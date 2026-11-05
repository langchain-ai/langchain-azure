"""Shell programs the dynamic-sessions backend sends, and their result parsing.

Pure string in, string out: nothing here performs I/O. Keeping the generated
programs separate lets them be exercised directly (see
`tests/unit_tests/dynamic_sessions/test_sessions_shell_programs.py`) rather
than only through a mocked HTTP layer, which is what let a wrong program ship
under passing tests.

Two rules hold throughout and must not be relaxed:

- Every interpolated path is bound once via `shlex.quote` and thereafter
  referenced only as `"$p"`. Never reflect a path into a double-quoted shell
  string: `$(...)`, backticks and `$VAR` in a filename are expanded remotely.
- Caller text (file content, edit strings, grep patterns) travels
  `shlex.quote`d or base64-encoded, never inline.

GNU userland (findutils `-printf`, `sort -z`, `xargs -0`) is assumed
throughout: every dynamic-sessions pool image ships it, and `glob` already
depends on it.

Filenames containing a newline are outside the contract of `ls`, `glob`, and
`grep`: all three parse line-oriented output, and such a name splits into
unrelated rows. The base class shares the limitation for `grep`; its
JSON-per-line `ls`/`glob` do not, but those run `python3`, which a Shell pool
image need not carry.
"""

from __future__ import annotations

import base64
import secrets
import shlex

# The probes cannot signal failure through exit status without also aborting
# the surrounding command list, so they print a sentinel instead.
ERR_SENTINEL = "__ERR__"
SEP_SENTINEL = "__SEP__"

# Exit statuses `write` and `edit` use to distinguish semantic outcomes without
# printing anything that could be confused with file content. Deliberately high
# and disjoint from the small codes shells and coreutils produce on their own
# (1-2 for redirect/`mkdir`/`awk` failures), so a generic tool failure cannot
# masquerade as "file exists" or "string not found".
EDIT_NOT_FOUND = 10
EDIT_AMBIGUOUS = 11
EDIT_MISSING_FILE = 12
EDIT_EMPTY_OLD = 13

WRITE_EXISTS = 10

# Linux caps one argv/environ string at MAX_ARG_STRLEN (128 KiB). Both `write`
# and `edit` inline their base64 payloads into the single `sh -c` program
# string -- for `edit` the two literals ride there together, and only the
# decoded values later become separate environment entries for awk -- so a
# program whose base64 nears the cap dies at exec with an unhelpful E2BIG deep
# in the pool (measured live: 64 KB of content passes, 120 KB fails). 90 KB of
# raw text is ~120 KB of base64, leaving margin for the surrounding program.
# Callers refuse larger payloads with an actionable error instead; `edit`
# budgets the sum of its two payloads, never the larger of them alone.
MAX_TRANSPORT_BYTES = 90_000

# Reassembles the whole file into one buffer and matches with index(), so an
# `old_string` spanning newlines matches and two occurrences on one line count
# as two. A line-oriented matcher (`grep -cF`) gets both wrong.
#
# `old`/`new` arrive via ENVIRON rather than `-v`: `-v` interprets escape
# sequences and would corrupt a backslash.
#
# `read()` hands the model LF-normalized content, so an `old_string` copied
# straight back out of a read is LF-only even when the file on disk is CRLF.
# Matching only the bytes as sent would break the protocol's ordinary
# read-then-edit path, so -- like `BaseSandbox.edit` -- try `old` as sent, then
# a CRLF variant, then an LF variant, and apply the same transform to `new` so
# the file keeps its own line-ending style. On a file of mixed endings
# `replace_all` reaches only the first matching style; a second edit gets the
# rest, which is also how the base behaves.
#
# The transforms are literal `index()`/`substr()` walks rather than `gsub`
# against "\r": a dynamic regex would put the escaping burden on every awk
# implementation the pool image might ship. CR and LF come from `sprintf`
# rather than escapes for the same reason.
_EDIT_AWK = (
    "function cnt(hay, needle,   n, i, rest) { "
    "n = 0; rest = hay; "
    "while ((i = index(rest, needle)) > 0) "
    "{ n++; rest = substr(rest, i + length(needle)) } "
    "return n } "
    "function rep(hay, needle, r, all,   out, i, done) { "
    'out = ""; done = 0; '
    "while ((i = index(hay, needle)) > 0) { "
    "if (!all && done) break; "
    "out = out substr(hay, 1, i - 1) r; "
    "hay = substr(hay, i + length(needle)); done = 1 } "
    "return out hay } "
    "function tocrlf(s) { return rep(rep(s, CR LF, LF, 1), LF, CR LF, 1) } "
    "function tolf(s) { return rep(s, CR LF, LF, 1) } "
    'BEGIN { CR = sprintf("%c", 13); LF = sprintf("%c", 10); '
    'old = ENVIRON["OLD"]; new = ENVIRON["NEW"]; '
    'ra = ENVIRON["RA"] + 0; tn = ENVIRON["TN"] + 0 } '
    '{ if (NR > 1) buf = buf "\\n"; buf = buf $0 } '
    "END { "
    f'if (old == "") {{ exit {EDIT_EMPTY_OLD} }} '
    "mo = old; mn = new; n = cnt(buf, mo); "
    "if (n == 0) { c = tocrlf(old); n = cnt(buf, c); "
    "if (n > 0) { mo = c; mn = tocrlf(new) } } "
    "if (n == 0) { c = tolf(old); n = cnt(buf, c); "
    "if (n > 0) { mo = c; mn = tolf(new) } } "
    f"if (n == 0) {{ exit {EDIT_NOT_FOUND} }} "
    f"if (n > 1 && !ra) {{ exit {EDIT_AMBIGUOUS} }} "
    'printf "%s", rep(buf, mo, mn, ra); '
    'if (tn) printf "\\n"; '
    'print n > "/dev/stderr" }'
)


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _quote_path(path: str) -> str:
    """Quote `path` for a `p=...` binding, safe as an awk operand.

    A relative path containing `=` (`V=1.txt`) parses as a variable assignment
    when awk reads it as a file operand, so relative paths gain a `./` prefix.
    Absolute paths -- the protocol's contract -- cannot contain `=` before a
    `/` in a way awk accepts as an identifier, and are left alone.
    """
    return shlex.quote(path if path.startswith("/") else "./" + path)


def make_done_token() -> str:
    """A fresh completion marker for one command.

    Random per invocation, not a fixed literal: the marker is compared against
    output that includes file content, so a constant one could be forged by a
    file whose bytes happen to contain it -- and would be silently stripped
    from that file's own content.
    """
    return f"__LCDONE{secrets.token_hex(8)}__"


def with_done(program: str, done: str) -> str:
    """Make `done` the program's final write, on every exit path.

    The dynamic-sessions data plane intermittently loses a process's last
    stdout write when that write is immediately followed by exit -- measured at
    1.6-4% against a live Shell pool. Trailing the real output with a marker
    means the output is no longer last (so it survives) and its absence
    identifies the runs where the tail was lost anyway.

    An EXIT trap rather than an appended command, so the early `exit 0` error
    branches emit it too. Verified not to disturb the exit status the callers
    read.
    """
    return f"trap 'printf %s {done}' EXIT; {program}"


def split_done(output: str, done: str) -> tuple[str, bool]:
    """Strip `done` from `output`, reporting whether it was there at all.

    Not a suffix check: `execute` returns stdout concatenated with stderr, and
    `edit` writes its occurrence count to stderr, so the marker ending stdout
    can sit mid-string.
    """
    before, sep, after = output.partition(done)
    return before + after, bool(sep)


def strip_partial_done(output: str, done: str) -> str:
    """Remove a marker fragment left when truncation cut `done` in half.

    When the byte cap lands inside the completion marker, `split_done` cannot
    find the full token and the leading fragment (`__LCDONE12ab...`) would leak
    into content handed to the model. Strip the longest output suffix that is a
    proper prefix of `done`. Only meaningful on truncated output: a complete
    marker is handled by `split_done`.
    """
    for length in range(len(done) - 1, 0, -1):
        if output.endswith(done[:length]):
            return output[:-length]
    return output


def build_ls_command(path: str) -> str:
    """List `path`, one bare name per line, directories suffixed with `/`.

    No `-L`: `BaseSandbox.ls` reports `is_dir` from
    `entry.is_dir(follow_symlinks=False)`, so a symlink to a directory is not a
    directory. Following it here would disagree with the base and with every
    other backend, and would also make `-p` mark the link with a trailing `/`.

    `2>/dev/null` on the listing itself: `ls` can still emit a per-entry
    diagnostic to stderr while listing the name on stdout, and `execute`
    concatenates the two streams, so an unsilenced diagnostic would parse as an
    entry. For the same reason the exit status is not consulted -- `ls` exits
    nonzero for exactly those per-entry diagnostics while the stdout listing
    remains complete; the probes above already caught the errors that mean "no
    listing at all".
    """
    return (
        f"p={shlex.quote(path)}; "
        f'if [ ! -e "$p" ]; then echo "{ERR_SENTINEL}path_not_found"; exit 0; fi; '
        f'if [ ! -d "$p" ]; then echo "{ERR_SENTINEL}not_a_directory"; exit 0; fi; '
        f'if [ ! -r "$p" ]; then echo "{ERR_SENTINEL}permission_denied"; exit 0; fi; '
        'ls -1ap "$p" 2>/dev/null'
    )


# The dynamic-sessions data plane caps each output stream at 4,096 bytes per
# execution and signals nothing when it cuts (measured live; the status stays
# "0" and no field marks the loss). Read windows are therefore fetched in
# base64 chunks sized so one chunk plus its bookkeeping and completion marker
# fit inside a single cap: 3,000 raw bytes become 4,000 base64 characters
# (`base64 | tr -d '\n'` rather than GNU's `-w0`, so the read programs stay
# runnable on BSD userland dev machines).
# base64, not raw text, because the transport is a JSON string: a chunk
# boundary landing inside a multi-byte character would otherwise be mangled
# by the service's replacement handling and break byte-exact reassembly.
# The data plane truncates each output stream at exactly this many bytes,
# silently and with a success status. Every read chunk is sized to land under
# it; `sessions` uses the same constant for its at-cap truncation heuristic.
SERVICE_OUTPUT_CAP = 4096

# What the first read command emits besides its base64 chunk: two counts, two
# separators with their newlines, the trailing `echo`, and the completion
# marker (`__LCDONE` + 16 hex + `__`, on its own line) -- about 53 bytes for a
# plausible file, reserved generously so a very large line count cannot eat
# into the chunk. A chunk must fit in what is left of the wire budget, or its
# base64 arrives cut and will not decode.
READ_BOOKKEEPING_BYTES = 96


def chunk_for_wire(room: int) -> int:
    """Largest raw byte count whose base64 fits in `room` bytes of output.

    base64 is 4 characters per 3 bytes, so this inverts that and floors to a
    whole group -- a partial group would arrive undecodable, which is the
    failure this exists to prevent. Returns 0 when nothing fits.
    """
    return max(0, (room // 4) * 3)


# Derived, not tuned by hand: the largest chunk that fits under the service
# cap alongside its bookkeeping. A caller whose `max_output_bytes` is lower
# gets a smaller chunk from the same formula, never a cut record.
READ_CHUNK_BYTES = chunk_for_wire(SERVICE_OUTPUT_CAP - READ_BOOKKEEPING_BYTES)


def _read_probes(file_path: str) -> str:
    """The `p=` binding plus the four probes every read-family command runs."""
    return (
        f"p={_quote_path(file_path)}; "
        f'if [ -d "$p" ]; then echo "{ERR_SENTINEL}is_directory"; exit 0; fi; '
        f'if [ ! -e "$p" ]; then echo "{ERR_SENTINEL}file_not_found"; exit 0; fi; '
        f'if [ ! -f "$p" ]; then echo "{ERR_SENTINEL}not_a_file"; exit 0; fi; '
        f'if [ ! -r "$p" ]; then echo "{ERR_SENTINEL}permission_denied"; exit 0; fi; '
    )


def build_read_command(
    file_path: str, *, start: int, end: int, take: int = READ_CHUNK_BYTES
) -> str:
    """Print line count, window byte count, and the window's first chunk.

    Output is three `SEP_SENTINEL`-separated records: the file's total line
    count, the byte size of the `start`..`end` window, and up to
    `take` bytes of the window as base64. The byte count is what lets
    the caller fetch the rest through `build_read_chunk_command` and know when
    it has everything -- the service cap otherwise eats the tail *and* the
    completion marker, making a capped window indistinguishable from a lost
    one.

    `awk 'END {print NR}'` rather than `wc -l`: `wc` counts newlines and so
    undercounts a file whose last line is unterminated.

    The `! -f` probe keeps awk away from FIFOs and devices, where reading
    blocks until the command timeout; `BaseSandbox` reports the same
    `not_a_file` code. `LC_ALL=C` pins awk to byte semantics so non-UTF-8
    content cannot provoke a locale warning on stderr, which `execute`
    concatenates into the output stream.

    `int()` on the bounds is load-bearing, not decorative: they are interpolated
    into the program, and coercion guarantees a non-integer cannot reach the
    shell even if the caller-side clamping above this is refactored away.
    """
    start = int(start)
    end = int(end)
    take = int(take)
    window = f"LC_ALL=C awk 'NR>={start} && NR<={end}' \"$p\""
    return (
        _read_probes(file_path)
        + "LC_ALL=C awk 'END {print NR}' \"$p\"; "
        + f'echo "{SEP_SENTINEL}"; '
        + f"{window} | wc -c; "
        + f'echo "{SEP_SENTINEL}"; '
        + f"{window} | head -c {take} | base64 | tr -d '\\n'; echo"
    )


def build_read_chunk_command(
    file_path: str, *, start: int, end: int, skip: int, take: int
) -> str:
    """One follow-up chunk of a read window, as base64.

    `skip` is a 0-based byte offset into the same awk window
    `build_read_command` measured (`tail -c +N` is 1-based, hence the +1).
    The probes re-run so a file deleted between chunks reports its sentinel
    rather than feeding awk nothing.
    """
    start, end, skip, take = int(start), int(end), int(skip), int(take)
    return (
        _read_probes(file_path)
        + f"LC_ALL=C awk 'NR>={start} && NR<={end}' \"$p\" | "
        + f"tail -c +{skip + 1} | head -c {take} | base64 | tr -d '\\n'; echo"
    )


def build_write_command(file_path: str, content: str) -> str:
    """Create `file_path`, refusing to overwrite. Outcome is the exit status.

    Only `WRITE_EXISTS` is semantic; any other nonzero status (`mkdir` or the
    redirect failing on permissions or a read-only mount) takes the callers'
    generic-failure branch.
    """
    return (
        f"p={shlex.quote(file_path)}; "
        f'if [ -e "$p" ]; then exit {WRITE_EXISTS}; fi; '
        'mkdir -p "$(dirname "$p")" || exit 2; '
        f"printf %s '{_b64(content)}' | base64 -d > \"$p\""
    )


def build_edit_command(
    file_path: str, old_string: str, new_string: str, *, replace_all: bool
) -> str:
    """Replace an exact substring, reporting the occurrence count on stderr.

    The shell variables are built with a trailing `X` sentinel that is then
    stripped, because command substitution discards trailing newlines -- without
    it an `old_string` ending in a newline could never match.

    The result lands via `cat > "$p"` rather than `mv`: `mv` would replace the
    inode, resetting the file's mode to `mktemp`'s 0600, breaking hard links,
    and replacing a symlink instead of its target. A failure of that final
    write exits 1 -- generic, deliberately outside the semantic codes -- with
    the temp file cleaned up either way.
    """
    return (
        f"p={_quote_path(file_path)}; "
        f'if [ ! -f "$p" ]; then exit {EDIT_MISSING_FILE}; fi; '
        f"OLD=$(printf %s '{_b64(old_string)}' | base64 -d; printf X); "
        "OLD=${OLD%X}; "
        f"NEW=$(printf %s '{_b64(new_string)}' | base64 -d; printf X); "
        "NEW=${NEW%X}; "
        'if [ -n "$(tail -c1 "$p")" ]; then TN=0; else TN=1; fi; '
        "TMPF=$(mktemp) || exit 1; "
        # LC_ALL=C: byte semantics for index()/substr(), and no locale warning
        # from gawk on non-UTF-8 content -- a warning would land on stderr and
        # poison the occurrence-count channel.
        f'OLD="$OLD" NEW="$NEW" RA={1 if replace_all else 0} TN="$TN" LC_ALL=C '
        f'awk \'{_EDIT_AWK}\' "$p" > "$TMPF"; '
        "STATUS=$?; "
        'if [ $STATUS -ne 0 ]; then rm -f "$TMPF"; exit $STATUS; fi; '
        'cat "$TMPF" > "$p" || { rm -f "$TMPF"; exit 1; }; '
        'rm -f "$TMPF"'
    )


def _has_wildcard(segment: str) -> bool:
    return any(c in segment for c in "*?[")


def _pinned_exclusions(segments: list[str]) -> str:
    """Hidden-file exclusions for a depth-pinned `-path` match.

    With the depth pinned, a path matching the primary can only place a hidden
    component at a wildcard-bearing segment position, so one `! -path` per such
    segment -- the segment replaced by `.*` -- excludes exactly the matches
    Python `glob` would refuse, while a segment that names a hidden entry
    (`.github`) keeps matching it. A literal segment needs no exclusion: it
    either names the component outright or cannot match it at all.
    """
    out = []
    for i, seg in enumerate(segments):
        if _has_wildcard(seg) and not seg.startswith("."):
            replaced = "/".join([*segments[:i], ".*", *segments[i + 1 :]])
            out.append(f" ! -path {shlex.quote('./' + replaced)}")
    return "".join(out)


def _anchored_exclusions(head: list[str], *, tail_names_hidden: bool) -> str:
    """Hidden-file exclusions for the recursive routes.

    Anchored below the deepest `head` segment that names a hidden entry (or the
    search root), because past a `**` the depth is unknown and `-path` cannot
    pin positions. When the final segment itself names a hidden entry
    (`**/.gitignore`), only *non-final* hidden components are forbidden -- the
    `/.*/*` shapes require a component after the hidden one -- so the named
    entry survives while the `**` still refuses to descend into hidden
    directories. Otherwise the plain shapes forbid a hidden component anywhere
    below the anchor, final position included.
    """
    named = [i for i, seg in enumerate(head) if seg.startswith(".")]
    anchor = "./" + "/".join(head[: named[-1] + 1]) if named else "."
    suffixes = ("/.*/*", "/*/.*/*") if tail_names_hidden else ("/.*", "/*/.*")
    return "".join(f" ! -path {shlex.quote(anchor + s)}" for s in suffixes)


def _one_plus_exclusions(parts: list[str], span_index: int) -> str:
    """Hidden-file exclusions for the unpinned `HEAD/*/TAIL` invocation.

    Same construction as `_pinned_exclusions` -- replace one wildcard segment
    with `.*` -- but the segment standing in for `**` may span several
    components, so it gets a second `*/.*` variant to catch a hidden component
    deeper inside the span. Wildcards in HEAD or TAIL may still cross `/` here
    (the route's documented approximation), and their exclusions inherit it.
    """
    out = []
    for i, seg in enumerate(parts):
        if not _has_wildcard(seg) or seg.startswith("."):
            continue
        variants = [[*parts[:i], ".*", *parts[i + 1 :]]]
        if i == span_index:
            variants.append([*parts[:i], "*/.*", *parts[i + 1 :]])
        for v in variants:
            out.append(f" ! -path {shlex.quote('./' + '/'.join(v))}")
    return "".join(out)


def _normalize_segments(rel_pattern: str) -> tuple[list[str], bool]:
    """Split a pattern into segments, dropping no-op components.

    `.` segments and empty segments (`a//b`) match nothing in `find -path`
    form but are no-ops in Python `glob`, so they are dropped before
    translation; matches are reported in the canonical spelling (`./x.txt`
    comes back as `x.txt`). A trailing `/` means "directories only" and is
    returned as a flag rather than a segment.
    """
    raw = rel_pattern.split("/")
    dirs_only = len(raw) > 1 and raw[-1] == ""
    return [s for s in raw if s not in (".", "")], dirs_only


def _glob_invocations(rel_pattern: str) -> list[tuple[str, str]]:
    """Translate a glob into `(find primaries, -printf format)` invocations.

    The protocol pins Python `glob` semantics: `*` matches within a single
    path segment, `**` matches any number of directories, and a wildcard never
    matches a leading dot -- a hidden entry is reached only by a segment that
    names it (`.env`, `.github/**`). `find -path` alone gets all of this wrong
    (its `*` crosses `/`), so common shapes are translated:

    - `NAME` (no `/`): `-mindepth 1 -maxdepth 1 -name NAME`, with `! -name
      '.*'` when NAME is a non-dot-leading wildcard.
    - `**` alone: `-mindepth 1` -- everything, recursively.
    - `**/TAIL` (TAIL a single segment): `-name TAIL` at any depth. The
      `-mindepth 1` keeps the search root out of its own results: the root's
      name is `.`, which a wildcard TAIL matches, and `-printf '%P'` renders
      it as an empty path.
    - `A/B/C` (no `**`): `-path './A/B/C'` pinned to exactly that many levels
      via `-mindepth`/`-maxdepth`. With the depth pinned, a `*` cannot swallow
      a `/` without changing the segment count, so per-segment matching falls
      out for free, and `_pinned_exclusions` can forbid a hidden component at
      exactly the wildcard positions.
    - `HEAD/**/TAIL` (one directory-position `**`): two invocations, merged
      and deduplicated -- the zero-directory shape `HEAD/TAIL` depth-pinned,
      plus `-path './HEAD/*/TAIL'` unpinned for one-or-more directories.
      Within the second, a wildcard inside HEAD or TAIL may still cross `/`;
      a documented approximation. For `HEAD/**` the zero-directory match is
      HEAD itself, reported with a trailing slash exactly as Python's `**`
      zero-expansion does.
    - A trailing `/` (`src/`, `*/`, `**/`) matches directories only
      (symlinks to directories included, as in Python), reported with the
      trailing slash.

    Anything else (multiple `**`, or `**` embedded inside a segment like
    `a**b`) falls back to a bare `-path` match -- fnmatch semantics where
    every star may cross `/`, with the hidden-entry exclusion anchored below
    the deepest segment naming a hidden entry. For `a**b` shapes that
    over-matches; a pattern with several `**` segments under-matches shallow
    paths, because fnmatch cannot make a `/`-adjacent star optional. A
    documented approximation.

    Three smaller divergences from Python `glob`, documented here rather than
    papered over: a backslash escapes the next wildcard character for `find`
    but is a literal character to Python; a pattern that is only no-op
    segments (`.`, `./`) returns nothing rather than the root itself; and
    `find -P` never *descends through* a symlink to a directory, where
    Python's glob does -- which also means a self-referential link cannot
    send `**` into an unbounded walk, as it does upstream.
    """
    segments, dirs_only = _normalize_segments(rel_pattern)
    if not segments:
        return []
    xd = " -xtype d" if dirs_only else ""
    fmt_plain = r"%P/\n" if dirs_only else r"%P\n"

    if segments == ["**"]:
        hide = _anchored_exclusions([], tail_names_hidden=False)
        return [(f"-mindepth 1{xd}{hide}", fmt_plain)]
    if not any("**" in seg for seg in segments):
        if len(segments) == 1:
            name = segments[0]
            hide = (
                " ! -name '.*'"
                if _has_wildcard(name) and not name.startswith(".")
                else ""
            )
            return [
                (
                    f"-mindepth 1 -maxdepth 1 -name {shlex.quote(name)}{xd}{hide}",
                    fmt_plain,
                )
            ]
        depth = len(segments)
        quoted = shlex.quote("./" + "/".join(segments))
        hide = _pinned_exclusions(segments)
        return [
            (f"-mindepth {depth} -maxdepth {depth} -path {quoted}{xd}{hide}", fmt_plain)
        ]
    if segments.count("**") == 1 and all(
        "**" not in seg for seg in segments if seg != "**"
    ):
        index = segments.index("**")
        head, tail = segments[:index], segments[index + 1 :]
        if not head and len(tail) == 1:
            hide = _anchored_exclusions([], tail_names_hidden=tail[0].startswith("."))
            return [(f"-mindepth 1 -name {shlex.quote(tail[0])}{xd}{hide}", fmt_plain)]
        zero_dir = head + tail
        depth = len(zero_dir)
        collapsed = shlex.quote("./" + "/".join(zero_dir))
        # `HEAD/**`'s zero-directory expansion is HEAD itself, directories
        # only, spelled `HEAD/` -- Python's `**` zero-expansion form.
        zero_xd = xd if tail else " -xtype d"
        zero_fmt = fmt_plain if tail else r"%P/\n"
        one_plus_parts = [*head, "*", *tail]
        one_plus = shlex.quote("./" + "/".join(one_plus_parts))
        return [
            (
                f"-mindepth {depth} -maxdepth {depth} -path {collapsed}"
                f"{zero_xd}{_pinned_exclusions(zero_dir)}",
                zero_fmt,
            ),
            (
                f"-path {one_plus}{xd}"
                f"{_one_plus_exclusions(one_plus_parts, len(head))}",
                fmt_plain,
            ),
        ]
    quoted = shlex.quote("./" + "/".join(segments))
    hide = _anchored_exclusions(
        segments, tail_names_hidden=segments[-1].startswith(".")
    )
    return [(f"-path {quoted}{xd}{hide}", fmt_plain)]


def _find_primaries(rel_pattern: str) -> list[str]:
    """The `find` primaries for `rel_pattern`, without the `-printf` formats.

    `grep`'s slash-glob route selects files from these primaries itself
    (`-type f -print0`), so only the expressions matter there.
    """
    return [primaries for primaries, _ in _glob_invocations(rel_pattern)]


def build_glob_command(search_path: str, rel_pattern: str) -> str:
    r"""Find paths under `search_path` matching `rel_pattern`, prefixed D:/F:.

    The probes report the same codes as `ls` for the same root failures --
    without them a `cd` into an existing-but-unreadable directory would be
    indistinguishable from a missing one. `-printf` is GNU findutils, which
    every dynamic-sessions pool image ships; BSD find has no equivalent.
    Multiple `find` invocations (see `_glob_invocations`) are merged through
    `sort -u`, which also dedupes the overlap between them.

    A match whose real path leaves the search root is dropped, matching
    `BaseSandbox.glob`, which resolves each candidate before keeping it.
    Without this a symlink pointing outside `path` is reported as a match, and
    rejecting `..` in the pattern would no longer be enough to keep results
    inside the root. A dangling symlink is dropped too (`[ ! -e ]`), as the
    base's `os.stat` filter does. The filename is only ever passed as a quoted
    `"$f"` after `--`, so a hostile name cannot reach the comparison as
    anything but data.

    The root is normalized to a trailing slash before the prefix test, and the
    root itself is allowed through. Comparing against a bare `"$root"/*` breaks
    for `glob()`'s documented default of `/`, where the pattern becomes `//*`
    and drops every candidate -- silently, since the command still exits 0.
    Keeping the separator in the prefix is also what stops `/mnt/database` from
    passing as a child of `/mnt/data`.

    Records are emitted with `printf`, never `echo`: POSIX `echo` (dash, ash)
    expands backslash escapes in its argument, so a filename containing `\\n`
    or `\\c` would be rewritten, split, or silently merged with the next
    record.
    """
    finds = (
        "; ".join(
            f"find . {primaries} -printf '{fmt}' 2>/dev/null"
            for primaries, fmt in _glob_invocations(rel_pattern)
        )
        or ":"
    )
    return (
        f"p={shlex.quote(search_path)}; "
        f'if [ ! -e "$p" ]; then echo "{ERR_SENTINEL}path_not_found"; exit 0; fi; '
        f'if [ ! -d "$p" ]; then echo "{ERR_SENTINEL}not_a_directory"; exit 0; fi; '
        # Both bits: `find` needs `r` to list and `cd` needs `x` to enter.
        f'if [ ! -r "$p" ] || [ ! -x "$p" ]; '
        f'then echo "{ERR_SENTINEL}permission_denied"; exit 0; fi; '
        'cd "$p" 2>/dev/null || exit 1; '
        "root=$(pwd -P) && "
        'case "$root" in */) rpfx="$root" ;; *) rpfx="$root/" ;; esac && '
        f"{{ {finds}; }} | sort -u | "
        "while IFS= read -r f; do "
        '[ -n "$f" ] || continue; '
        '[ -e "$f" ] || continue; '
        'rp=$(readlink -f -- "$f" 2>/dev/null) || continue; '
        'if [ "$rp" != "$root" ]; then '
        'case "$rp" in "$rpfx"*) ;; *) continue ;; esac; fi; '
        # `-d` follows the link deliberately: base `glob` classifies with
        # `os.path.isdir`, which follows, while base `ls` uses
        # `follow_symlinks=False`. The asymmetry is upstream's; mirror it.
        'if [ -d "$f" ]; then printf \'D:%s\\n\' "$f"; '
        "else printf 'F:%s\\n' \"$f\"; fi; "
        "done"
    )


def build_grep_command(
    search_path: str,
    pattern: str,
    *,
    include_glob: str | None,
    max_count: int | None,
) -> str:
    """Search `search_path` for the literal `pattern`, `grep -F` style.

    Mirrors `BaseSandbox`'s grep contract without its `python3 -c` fallback,
    which a Shell pool image need not carry:

    - No glob, or a basename glob (no `/`): GNU `grep -rHnFZ`, with
      `--include` when a glob was given.
    - A slash-containing glob: the same `find` translation `glob` uses
      enumerates matching files (`-type f`), and the batch is handed to
      `grep -HnFZ` via NUL-separated `xargs`.

    Output records are `path NUL line:text`, matching what the caller parses.
    `head -n max_count+1` stops the search one record past the cap so the
    parser can distinguish "exactly at the cap" from "capped early".

    The existence/readability probes make the common failures explicit; after
    them, grep's own exit status is discarded (`|| true`) because exit 1 only
    means "no matches".
    """
    probes = (
        f"p={shlex.quote(search_path)}; "
        f'if [ ! -e "$p" ]; then echo "{ERR_SENTINEL}path_not_found"; exit 0; fi; '
        f'if [ ! -r "$p" ]; then echo "{ERR_SENTINEL}permission_denied"; exit 0; fi; '
    )
    cap = f" | head -n {int(max_count) + 1}" if max_count is not None else ""
    quoted_pattern = shlex.quote(pattern)
    if include_glob is not None and "/" in include_glob:
        finds = "; ".join(
            f"find . {primaries} -type f -print0 2>/dev/null"
            for primaries in _find_primaries(include_glob.lstrip("/"))
        )
        return (
            probes + f'if [ ! -d "$p" ]; then echo "{ERR_SENTINEL}not_a_directory"; '
            "exit 0; fi; "
            'cd "$p" 2>/dev/null || exit 1; '
            f"{{ {finds}; }} | sort -zu | "
            f"xargs -0 -r grep -HnFZ -e {quoted_pattern} -- 2>/dev/null"
            f"{cap}; true"
        )
    include = (
        f"--include={shlex.quote(include_glob)} " if include_glob is not None else ""
    )
    return (
        probes + f'grep -rHnFZ {include}-e {quoted_pattern} "$p" 2>/dev/null{cap}; true'
    )


def error_code(line: str) -> str | None:
    """Return the sentinel error code carried by `line`, if any."""
    stripped = line.strip()
    if stripped.startswith(ERR_SENTINEL):
        return stripped[len(ERR_SENTINEL) :]
    return None


def has_traversal(rel_pattern: str) -> bool:
    """Whether `rel_pattern` contains a `..` segment.

    Rejected outright rather than quoted, mirroring the base implementation, so
    a pattern cannot walk out of the search root.
    """
    return any(seg == ".." for seg in rel_pattern.replace("\\", "/").split("/"))
