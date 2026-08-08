"""Run `SessionsBashBackend`'s generated shell programs under a real shell.

The other backend suites stub `execute()`, so they check how a result is built
from a *manufactured* command output and never run the program itself. That
cannot catch a wrong shell program -- `edit()` shipped with `grep -cF`
occurrence counting that silently edited ambiguous matches, and `write()`
shipped with an unquoted `$(dirname ...)`, both under passing tests.

Here `execute()` runs the command through `/bin/sh` against a temp directory,
so these tests exercise the same text the session pool would. They are still
unit tests: no network, no Azure.
"""

from __future__ import annotations

import base64
import fnmatch
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest

pytest.importorskip("deepagents")

from deepagents.backends.protocol import (  # noqa: E402
    ExecuteResponse,
    ReadResult,
)
from deepagents.backends.sandbox import (  # noqa: E402
    TRUNCATION_MSG,
    BaseSandbox,
)

from langchain_azure_container_apps.dynamic_sessions.backends import (  # noqa: E402
    SessionsBashBackend,
    _shell,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


#: Anchored at the search root -- the shape the recursive routes use for a
#: pattern naming nothing hidden.
AT_ROOT = "! -path './.*' ! -path './*/.*'"


class LocalShellBackend(SessionsBashBackend):
    """A backend whose `execute()` runs locally instead of over HTTP."""

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        completed = subprocess.run(  # noqa: S603
            ["/bin/sh", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout or 30,
        )
        return ExecuteResponse(
            output=completed.stdout + completed.stderr,
            exit_code=completed.returncode,
            truncated=False,
        )


@pytest.fixture
def backend() -> LocalShellBackend:
    return LocalShellBackend(
        pool_management_endpoint="https://example.test/pool",
        access_token_provider=lambda: "token",
    )


@pytest.fixture
def workdir() -> Iterator[str]:
    with tempfile.TemporaryDirectory() as d:
        yield d


def _write(workdir: str, name: str, content: str) -> str:
    path = os.path.join(workdir, name)
    Path(path).write_text(content)
    return path


class TestEditOccurrenceCounting:
    def test_two_occurrences_on_one_line_are_ambiguous(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        # `grep -cF` counts lines, so this used to report one match and edit it.
        path = _write(workdir, "f.txt", "apple apple\n")
        result = backend.edit(path, "apple", "PEAR")
        assert result.error is not None
        assert "appears multiple times" in result.error
        assert Path(path).read_text() == "apple apple\n"

    def test_two_occurrences_on_one_line_with_replace_all(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        path = _write(workdir, "f.txt", "apple apple\n")
        result = backend.edit(path, "apple", "PEAR", replace_all=True)
        assert result.occurrences == 2
        assert Path(path).read_text() == "PEAR PEAR\n"

    def test_counts_occurrences_across_lines(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        path = _write(workdir, "f.txt", "a x\nb x\n")
        result = backend.edit(path, "x", "Y", replace_all=True)
        assert result.occurrences == 2
        assert Path(path).read_text() == "a Y\nb Y\n"


class TestEditMultiline:
    def test_multiline_old_string_matches(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        # A line-oriented matcher can never see this as one substring.
        path = _write(workdir, "f.txt", "Line 1\nLine 2\nLine 3\n")
        result = backend.edit(path, "Line 1\nLine 2", "REPLACED")
        assert result.error is None
        assert result.occurrences == 1
        assert Path(path).read_text() == "REPLACED\nLine 3\n"

    def test_replacement_may_introduce_newlines(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        path = _write(workdir, "f.txt", "abc\n")
        backend.edit(path, "b", "1\n2")
        assert Path(path).read_text() == "a1\n2c\n"

    def test_old_string_may_end_with_a_newline(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        # Command substitution strips trailing newlines without the sentinel.
        path = _write(workdir, "f.txt", "a\nb\n")
        backend.edit(path, "a\n", "Z")
        assert Path(path).read_text() == "Zb\n"


class TestEditContentPreservation:
    def test_backslashes_survive(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        # `awk -v` expands escapes; ENVIRON does not.
        path = _write(workdir, "f.txt", "a\\nb\n")
        backend.edit(path, "\\n", "X")
        assert Path(path).read_text() == "aXb\n"

    def test_missing_trailing_newline_is_not_added(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        path = _write(workdir, "f.txt", "abc")
        backend.edit(path, "b", "X")
        assert Path(path).read_text() == "aXc"

    def test_blank_lines_are_preserved(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        path = _write(workdir, "f.txt", "a\n\n\nb\n")
        backend.edit(path, "b", "B")
        assert Path(path).read_text() == "a\n\n\nB\n"

    def test_shell_metacharacters_in_content_are_inert(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        path = _write(workdir, "f.txt", "x $(id) `id` ${HOME} y\n")
        backend.edit(path, "x", "X")
        assert Path(path).read_text() == "X $(id) `id` ${HOME} y\n"


class TestEditErrors:
    def test_string_not_found(self, backend: LocalShellBackend, workdir: str) -> None:
        path = _write(workdir, "f.txt", "abc\n")
        assert "String not found" in str(backend.edit(path, "zzz", "X").error)

    def test_missing_file(self, backend: LocalShellBackend, workdir: str) -> None:
        path = os.path.join(workdir, "nope.txt")
        assert "not found" in str(backend.edit(path, "a", "X").error)

    def test_empty_old_string(self, backend: LocalShellBackend, workdir: str) -> None:
        path = _write(workdir, "f.txt", "abc\n")
        assert "must not be empty" in str(backend.edit(path, "", "X").error)


class TestWriteQuoting:
    def test_writes_content_verbatim(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        path = os.path.join(workdir, "f.txt")
        assert backend.write(path, "hello\nworld\n").path == path
        assert Path(path).read_text() == "hello\nworld\n"

    def test_refuses_to_overwrite(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        path = _write(workdir, "f.txt", "original")
        result = backend.write(path, "replacement")
        assert result.error is not None
        assert "already exists" in result.error
        assert Path(path).read_text() == "original"

    def test_creates_parent_directories_with_spaces(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        # `mkdir -p $(dirname ...)` unquoted word-splits this into two dirs.
        path = os.path.join(workdir, "a dir", "nested", "f.txt")
        assert backend.write(path, "x").path == path
        assert Path(path).read_text() == "x"

    @pytest.mark.parametrize(
        "name",
        [
            "with space.txt",
            "with'quote.txt",
            'with"doublequote.txt',
            "with$(id).txt",
            "with`id`.txt",
            "with${HOME}.txt",
            "with;semicolon.txt",
            "with|pipe.txt",
            "with*glob.txt",
        ],
    )
    def test_hostile_filenames_are_not_interpreted(
        self, backend: LocalShellBackend, workdir: str, name: str
    ) -> None:
        path = os.path.join(workdir, name)
        assert backend.write(path, "payload").path == path
        assert Path(path).read_text() == "payload"
        assert sorted(os.listdir(workdir)) == [name]

    def test_existing_hostile_filename_reports_conflict(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        # The old error branch echoed the raw path inside double quotes, so a
        # `$(...)` in the name was expanded by the remote shell.
        path = _write(workdir, "$(touch pwned).txt", "original")
        result = backend.write(path, "replacement")
        assert "already exists" in str(result.error)
        assert not os.path.exists(os.path.join(workdir, "pwned"))


class TestReadAndLs:
    def test_read_round_trips_written_content(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        path = _write(workdir, "f.txt", "one\ntwo\nthree\n")
        result = backend.read(path)
        assert result.file_data["content"] == "one\ntwo\nthree"
        assert result.total_lines == 3

    def test_read_pagination(self, backend: LocalShellBackend, workdir: str) -> None:
        path = _write(workdir, "f.txt", "".join(f"L{i}\n" for i in range(10)))
        result = backend.read(path, offset=3, limit=2)
        assert result.file_data["content"] == "L3\nL4"
        assert result.next_offset == 5

    def test_ls_returns_absolute_posix_paths(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        _write(workdir, "f.txt", "x")
        os.mkdir(os.path.join(workdir, "sub"))
        entries = backend.ls(workdir).entries
        assert entries is not None
        by_path = {e["path"]: e["is_dir"] for e in entries}
        assert by_path == {
            f"{workdir}/f.txt": False,
            f"{workdir}/sub": True,
        }


def _missing_gnu_features() -> list[str]:
    """GNU userland features the generated programs use that BSD lacks.

    The pool image is Linux, so the programs may assume these; a macOS
    maintainer running the suite locally cannot. Probing each one by name
    reports *what* is missing rather than skipping under a vague reason.
    """
    probes = {
        "find -printf": "find . -maxdepth 0 -printf ''",
        "sort -z": "printf '' | sort -z",
        "xargs -0": "printf '' | xargs -0 true",
    }
    return [
        name
        for name, probe in probes.items()
        if subprocess.run(  # noqa: S603
            ["/bin/sh", "-c", f"{probe} 2>/dev/null"], capture_output=True
        ).returncode
        != 0
    ]


_MISSING_GNU = _missing_gnu_features()

# Applied to every class that runs a generated program through the local shell.
requires_gnu_userland = pytest.mark.skipif(
    bool(_MISSING_GNU),
    reason=(
        "the generated programs assume GNU userland, which the session pool "
        f"image ships; this host lacks {', '.join(_MISSING_GNU)}. These run on "
        "Linux, and against the live pool in the integration suite."
    ),
)


class TestReadWindow:
    """Windows the protocol rejects if `end_line` lands below `start_line`."""

    @pytest.mark.parametrize(
        ("content", "expected", "total"),
        [
            ("\n", "", 1),
            ("\n\n", "\n", 2),
            ("\na\n", "\na", 2),
            ("hi\n", "hi", 1),
            ("hi", "hi", 1),
        ],
    )
    def test_blank_lines_produce_a_valid_window(
        self,
        backend: LocalShellBackend,
        workdir: str,
        content: str,
        expected: str,
        total: int,
    ) -> None:
        # A one-newline file counted as zero returned lines, producing
        # end_line=0 against start_line=1.
        path = _write(workdir, "f.txt", content)
        result = backend.read(path)
        assert result.error is None
        assert result.file_data["content"] == expected
        assert result.total_lines == total
        assert 1 <= result.start_line <= result.end_line


class TestGlob:
    @requires_gnu_userland
    def test_matches_are_relative_to_the_search_root(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        os.mkdir(os.path.join(workdir, "pkg"))
        _write(workdir, "pkg/mod.py", "x")
        result = backend.glob("**/*.py", workdir)
        assert result.matches == [{"path": "pkg/mod.py", "is_dir": False}]

    def test_missing_root_is_an_error_not_an_empty_match_set(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        # A failed `cd` used to look identical to "directory exists but empty".
        result = backend.glob("*.py", os.path.join(workdir, "does-not-exist"))
        assert result.matches is None
        assert result.error is not None


class TestCompletionMarker:
    """The marker must be the last write on every exit path, and inert.

    The data plane loses a process's final stdout write when that write is
    immediately followed by exit, which silently emptied `ls` listings. The
    marker works only if it is genuinely last -- including out of the early
    `exit 0` error branches -- and only if it leaves the exit status alone,
    which is how `write` and `edit` report their outcome.
    """

    def _run(self, program: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            ["/bin/sh", "-c", program], capture_output=True, text=True, timeout=30
        )

    def test_emitted_after_normal_output(self, workdir: str) -> None:
        _write(workdir, "a.txt", "x")
        done = _shell.make_done_token()
        out = self._run(_shell.with_done(_shell.build_ls_command(workdir), done)).stdout
        assert out.endswith(done)
        assert "a.txt" in out

    def test_emitted_out_of_an_early_error_exit(self, workdir: str) -> None:
        # `ls` of a missing path prints a sentinel and `exit 0`s before
        # reaching any appended command; only a trap survives that.
        done = _shell.make_done_token()
        program = _shell.build_ls_command(os.path.join(workdir, "nope"))
        result = self._run(_shell.with_done(program, done))
        assert result.stdout.endswith(done)
        assert _shell.ERR_SENTINEL in result.stdout

    @pytest.mark.parametrize("status", [0, 1, 2, 4])
    def test_exit_status_is_preserved(self, status: int) -> None:
        done = _shell.make_done_token()
        result = self._run(_shell.with_done(f"exit {status}", done))
        assert result.returncode == status
        assert result.stdout == done

    def test_write_refusal_keeps_its_status_and_marker(self, workdir: str) -> None:
        _write(workdir, "a.txt", "x")
        done = _shell.make_done_token()
        program = _shell.build_write_command(os.path.join(workdir, "a.txt"), "y")
        result = self._run(_shell.with_done(program, done))
        assert result.returncode == _shell.WRITE_EXISTS
        assert result.stdout.endswith(done)

    def test_token_is_fresh_per_call(self) -> None:
        """Fixed markers could be forged by file content, which read() echoes."""
        assert len({_shell.make_done_token() for _ in range(100)}) == 100

    def test_split_done_finds_a_marker_followed_by_stderr(self) -> None:
        # `execute` returns stdout + stderr, and `edit` writes its count to
        # stderr, so the marker ending stdout lands mid-string.
        done = _shell.make_done_token()
        body, complete = _shell.split_done(f"out{done}2", done)
        assert (body, complete) == ("out2", True)

    def test_split_done_reports_a_missing_marker(self) -> None:
        done = _shell.make_done_token()
        assert _shell.split_done("out", done) == ("out", False)


@requires_gnu_userland
class TestGlobSemantics:
    """The protocol pins Python-glob semantics: `*` stays within one path
    segment, `**` recurses. `find -path` alone gets both wrong."""

    @pytest.fixture
    def tree(self, workdir: str) -> str:
        for rel in (
            "top.py",
            "note.txt",
            "a/deep.py",
            "a/b/deeper.py",
            "src/main.py",
            "src/sub/util.py",
        ):
            path = Path(workdir, rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x\n")
        return workdir

    def _paths(self, backend: LocalShellBackend, pattern: str, tree: str) -> set[str]:
        result = backend.glob(pattern, path=tree)
        assert result.error is None, result.error
        return {m["path"] for m in result.matches or []}

    def test_star_does_not_cross_directories(
        self, backend: LocalShellBackend, tree: str
    ) -> None:
        # `find -path './*.py'` used to match a/deep.py too.
        assert self._paths(backend, "*.py", tree) == {"top.py"}

    def test_doublestar_matches_every_depth_including_root(
        self, backend: LocalShellBackend, tree: str
    ) -> None:
        # `find -path './**/*.py'` used to miss top.py.
        assert self._paths(backend, "**/*.py", tree) == {
            "top.py",
            "a/deep.py",
            "a/b/deeper.py",
            "src/main.py",
            "src/sub/util.py",
        }

    def test_single_level_slash_pattern(
        self, backend: LocalShellBackend, tree: str
    ) -> None:
        assert self._paths(backend, "src/*.py", tree) == {"src/main.py"}

    def test_head_doublestar_tail(self, backend: LocalShellBackend, tree: str) -> None:
        assert self._paths(backend, "src/**/*.py", tree) == {
            "src/main.py",
            "src/sub/util.py",
        }

    def test_bare_doublestar_matches_everything(
        self, backend: LocalShellBackend, tree: str
    ) -> None:
        paths = self._paths(backend, "**", tree)
        assert "top.py" in paths
        assert "a/b/deeper.py" in paths
        assert "src" in paths  # directories match too

    def test_star_matches_directories(
        self, backend: LocalShellBackend, tree: str
    ) -> None:
        result = backend.glob("*", path=tree)
        dirs = {m["path"] for m in result.matches or [] if m["is_dir"]}
        assert {"a", "src"} <= dirs

    def test_exotic_patterns_take_the_fnmatch_fallback(
        self, backend: LocalShellBackend, tree: str
    ) -> None:
        # Multiple `**` take the documented fnmatch approximation: each star
        # crosses `/` but none is optional, so only paths deep enough to feed
        # every slash match. Pinned here so the deviation stays deliberate.
        paths = self._paths(backend, "**/**/*.py", tree)
        assert paths == {"a/b/deeper.py", "src/sub/util.py"}


@requires_gnu_userland
class TestGrepPrograms:
    @pytest.fixture
    def tree(self, workdir: str) -> str:
        for rel, content in (
            ("app.py", "import os\nneedle here\n"),
            ("doc.txt", "needle here too\n"),
            ("src/lib.py", "no match\nneedle deep\n"),
            ("src/with space.py", "needle spaced\n"),
        ):
            path = Path(workdir, rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        return workdir

    def test_matches_carry_path_line_and_text(
        self, backend: LocalShellBackend, tree: str
    ) -> None:
        result = backend.grep("needle", path=tree)
        assert result.error is None
        found = {(m["path"], m["line"]) for m in result.matches or []}
        assert (f"{tree}/app.py", 2) in found
        assert (f"{tree}/doc.txt", 1) in found
        assert (f"{tree}/src/lib.py", 2) in found

    def test_basename_glob_filters_files(
        self, backend: LocalShellBackend, tree: str
    ) -> None:
        result = backend.grep("needle", path=tree, glob="*.py")
        paths = {m["path"] for m in result.matches or []}
        assert f"{tree}/doc.txt" not in paths
        assert f"{tree}/app.py" in paths

    def test_slash_glob_routes_through_find(
        self, backend: LocalShellBackend, tree: str
    ) -> None:
        # `grep --include` matches basenames only; a slash glob used to route
        # through a `python3 -c` wrapper a Shell pool may not carry.
        result = backend.grep("needle", path=tree, glob="src/**/*.py")
        assert result.error is None
        paths = {m["path"] for m in result.matches or []}
        assert paths == {f"{tree}/src/lib.py", f"{tree}/src/with space.py"}

    def test_max_count_truncates_and_flags(
        self, backend: LocalShellBackend, tree: str
    ) -> None:
        result = backend.grep("needle", path=tree, max_count=1)
        assert result.error is None
        assert len(result.matches or []) == 1
        assert result.truncated is True

    def test_exactly_at_the_cap_is_complete(
        self, backend: LocalShellBackend, tree: str
    ) -> None:
        result = backend.grep("import os", path=tree, max_count=5)
        assert len(result.matches or []) == 1
        assert result.truncated is False

    def test_missing_root_is_reported(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        result = backend.grep("needle", path=f"{workdir}/nope")
        assert result.matches is None
        assert result.error == f"Path '{workdir}/nope': path_not_found"

    def test_root_may_be_a_single_file(
        self, backend: LocalShellBackend, tree: str
    ) -> None:
        result = backend.grep("needle", path=f"{tree}/app.py")
        assert [(m["line"], m["text"]) for m in result.matches or []] == [
            (2, "needle here")
        ]

    def test_slash_glob_against_a_file_root_is_an_error(
        self, backend: LocalShellBackend, tree: str
    ) -> None:
        result = backend.grep("needle", path=f"{tree}/app.py", glob="src/*.py")
        assert result.matches is None
        assert result.error is not None
        assert "not_a_directory" in result.error

    def test_pattern_with_quotes_is_inert(
        self, backend: LocalShellBackend, tree: str
    ) -> None:
        result = backend.grep("ne'ed\"le; touch /tmp/pwned", path=tree)
        assert result.error is None
        assert result.matches == []

    def test_no_matches_is_empty_not_an_error(
        self, backend: LocalShellBackend, tree: str
    ) -> None:
        result = backend.grep("absent-string", path=tree)
        assert result.error is None
        assert result.matches == []


class TestEditPreservesFileIdentity:
    def test_mode_and_inode_survive_an_edit(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        # `mv` from mktemp used to replace the inode and reset 0755 to 0600.
        path = _write(workdir, "run.sh", "echo old\n")
        os.chmod(path, 0o755)
        inode = os.stat(path).st_ino
        result = backend.edit(path, "old", "new")
        assert result.error is None
        assert Path(path).read_text() == "echo new\n"
        assert os.stat(path).st_mode & 0o777 == 0o755
        assert os.stat(path).st_ino == inode


class TestTransportBudgetFitsTheKernelLimit:
    """`MAX_TRANSPORT_BYTES` budgets raw text, but what the kernel caps is the
    generated program. Raising the budget without rechecking this is exactly
    what would put `edit` back over MAX_ARG_STRLEN and into an in-pool E2BIG.
    """

    MAX_ARG_STRLEN = 128 * 1024
    DEEP_PATH = "/mnt/data/" + "dir/" * 40 + "file.txt"

    def test_worst_case_allowed_edit_program_fits(self) -> None:
        # The two payloads base64-pad separately, so an even split is worst.
        half = _shell.MAX_TRANSPORT_BYTES // 2
        command = _shell.with_done(
            _shell.build_edit_command(
                self.DEEP_PATH,
                "a" * half,
                "b" * (_shell.MAX_TRANSPORT_BYTES - half),
                replace_all=False,
            ),
            _shell.make_done_token(),
        )
        assert len(command.encode("utf-8")) < self.MAX_ARG_STRLEN

    def test_worst_case_allowed_write_program_fits(self) -> None:
        command = _shell.build_write_command(
            self.DEEP_PATH, "a" * _shell.MAX_TRANSPORT_BYTES
        )
        assert len(command.encode("utf-8")) < self.MAX_ARG_STRLEN


@requires_gnu_userland
class TestLsDiagnostics:
    def test_dangling_symlink_does_not_fabricate_entries(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        # `ls -L` prints "cannot access" to stderr for a dangling symlink;
        # unsilenced, that diagnostic parsed as a directory entry.
        _write(workdir, "real.txt", "x\n")
        os.symlink(f"{workdir}/missing", f"{workdir}/dangling")
        result = backend.ls(workdir)
        assert result.error is None
        names = {e["path"] for e in result.entries or []}
        assert names == {f"{workdir}/real.txt", f"{workdir}/dangling"}

    def test_err_sentinel_filename_does_not_poison_the_listing(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        _write(workdir, "__ERR__weird.txt", "x\n")
        _write(workdir, "normal.txt", "x\n")
        result = backend.ls(workdir)
        assert result.error is None
        assert {e["path"] for e in result.entries or []} == {
            f"{workdir}/__ERR__weird.txt",
            f"{workdir}/normal.txt",
        }


class TestShellHelpers:
    """Pure-string helpers with branches the end-to-end runs cannot pin."""

    def test_strip_partial_done_removes_a_bisected_marker(self) -> None:
        done = _shell.make_done_token()
        assert _shell.strip_partial_done("data" + done[:10], done) == "data"

    def test_strip_partial_done_leaves_unrelated_tails(self) -> None:
        done = _shell.make_done_token()
        assert _shell.strip_partial_done("data__OTHER", done) == "data__OTHER"

    def test_find_primaries_single_doublestar_with_multiseg_tail(self) -> None:
        primaries = _shell._find_primaries("**/src/*.py")
        assert len(primaries) == 2
        assert any("-mindepth 2 -maxdepth 2" in p for p in primaries)

    # The embedded-`**` fallback (`a**b`) is pinned in
    # `TestFindPrimaries.test_pattern_translation`; one copy of that literal is
    # enough, and two drift apart on every change to the exclusion shape.


class TestFindPrimaries:
    """`_find_primaries` is pure, so it is tested without a shell.

    The classes above prove the *behaviour* of these primaries by running
    `find`, which needs GNU userland and therefore skips on macOS. Pinning the
    translation itself here keeps the pattern-shape logic covered on every
    platform -- it is string-in, string-out and has no reason to need a shell.
    """

    @pytest.mark.parametrize(
        ("pattern", "expected"),
        [
            # A wildcard basename cannot match a hidden name.
            ("*.py", ["-mindepth 1 -maxdepth 1 -name '*.py' ! -name '.*'"]),
            # All-literal patterns need no exclusion: a literal either names
            # the component outright or cannot match it at all.
            ("a/b.py", ["-mindepth 2 -maxdepth 2 -path ./a/b.py"]),
            (".env", ["-mindepth 1 -maxdepth 1 -name .env"]),
            ("**", [f"-mindepth 1 {AT_ROOT}"]),
            # `-mindepth 1` keeps the root, whose `%P` is empty, out of results.
            ("**/*.py", [f"-mindepth 1 -name '*.py' {AT_ROOT}"]),
            # A hidden *final* segment stays reachable while the `**` still
            # refuses hidden directories: only non-final hidden components are
            # forbidden (the `/.*/*` shapes need a component after the dot).
            (
                "**/.gitignore",
                [
                    "-mindepth 1 -name .gitignore ! -path './.*/*' ! -path './*/.*/*'",
                ],
            ),
            (
                "src/**/x.py",
                [
                    "-mindepth 2 -maxdepth 2 -path ./src/x.py",
                    "-path './src/*/x.py' "
                    "! -path './src/.*/x.py' ! -path './src/*/.*/x.py'",
                ],
            ),
            ("a**b", [f"-path './a**b' {AT_ROOT}"]),
            ("**/x/**", [f"-path './**/x/**' {AT_ROOT}"]),
            (
                ".github/**/x.yml",
                [
                    "-mindepth 2 -maxdepth 2 -path ./.github/x.yml",
                    "-path './.github/*/x.yml' "
                    "! -path './.github/.*/x.yml' "
                    "! -path './.github/*/.*/x.yml'",
                ],
            ),
            # Wildcard segments at a pinned depth are excluded positionally:
            # the hidden variant replaces exactly the wildcard segment.
            (
                "*/.env",
                [
                    "-mindepth 2 -maxdepth 2 -path './*/.env' ! -path './.*/.env'",
                ],
            ),
            # `.` segments are honored for matching but dropped from the
            # translation; a trailing slash matches directories only.
            ("./*.txt", ["-mindepth 1 -maxdepth 1 -name '*.txt' ! -name '.*'"]),
            ("src/", ["-mindepth 1 -maxdepth 1 -name src -xtype d"]),
        ],
    )
    def test_pattern_translation(self, pattern: str, expected: list[str]) -> None:
        assert _shell._find_primaries(pattern) == expected

    def test_a_pattern_of_only_noop_segments_yields_nothing(self) -> None:
        """`.`/`./` normalize to nothing; the command degrades to a no-op
        rather than reporting the root (a documented divergence -- Python's
        `glob('.')` returns the root itself)."""
        assert _shell._find_primaries(".") == []
        command = _shell.build_glob_command("/tmp", ".")
        assert "find" not in command


class TestEditLineEndings:
    """`read` normalizes CRLF to LF, so `edit` must accept LF against a CRLF
    file -- otherwise the protocol's ordinary read-then-edit path is broken."""

    def test_lf_old_string_matches_a_crlf_file(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        path = os.path.join(workdir, "f.txt")
        Path(path).write_bytes(b"alpha\r\nbeta\r\n")
        content = backend.read(path).file_data["content"]  # type: ignore[index]
        assert content == "alpha\nbeta"
        assert backend.edit(path, content, "one\ntwo").error is None
        # The file's own CRLF style survives: `new` took the same transform.
        assert Path(path).read_bytes() == b"one\r\ntwo\r\n"

    def test_lf_file_is_untouched_by_the_crlf_fallback(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        path = os.path.join(workdir, "f.txt")
        Path(path).write_bytes(b"alpha\nbeta\n")
        assert backend.edit(path, "alpha\nbeta", "one\ntwo").error is None
        assert Path(path).read_bytes() == b"one\ntwo\n"

    def test_crlf_old_string_still_matches_verbatim(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        """The as-sent attempt runs first, so an explicit CRLF string works."""
        path = os.path.join(workdir, "f.txt")
        Path(path).write_bytes(b"alpha\r\nbeta\r\n")
        assert backend.edit(path, "alpha\r\nbeta", "x").error is None
        assert Path(path).read_bytes() == b"x\r\n"


class TestLsFidelity:
    """`ls` must return the bytes of a name, not a tidied version of it."""

    def test_leading_and_trailing_spaces_survive(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        # `strip()` on the whole output ate the leading space of the name that
        # sorts first and the trailing space of the one that sorts last, so the
        # returned paths named files that do not exist.
        for name in (" leading.txt", "middle.txt", "z trailing "):
            Path(workdir, name).write_text("x")
        entries = backend.ls(workdir).entries
        assert entries is not None
        assert sorted(os.path.basename(e["path"]) for e in entries) == [
            " leading.txt",
            "middle.txt",
            "z trailing ",
        ]

    def test_directory_symlink_is_not_a_directory(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        """`BaseSandbox.ls` uses `is_dir(follow_symlinks=False)`."""
        os.mkdir(os.path.join(workdir, "realdir"))
        os.symlink(os.path.join(workdir, "realdir"), os.path.join(workdir, "link"))
        entries = backend.ls(workdir).entries
        assert entries is not None
        by_name = {os.path.basename(e["path"]): e["is_dir"] for e in entries}
        assert by_name == {"realdir": True, "link": False}


@requires_gnu_userland
class TestGlobHiddenFiles:
    """Python `glob` does not let a wildcard match a leading dot; `find` does
    unless told otherwise. The mismatch changes which files an agent edits."""

    def _paths(self, backend: LocalShellBackend, pattern: str, root: str) -> set[str]:
        matches = backend.glob(pattern, root).matches
        return {m["path"] for m in matches or []}

    def test_wildcard_skips_hidden_entries(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        _write(workdir, ".hidden", "x")
        _write(workdir, "visible", "x")
        assert self._paths(backend, "*", workdir) == {"visible"}

    def test_recursive_wildcard_skips_hidden_directories(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        os.mkdir(os.path.join(workdir, ".git"))
        _write(workdir, ".git/config.py", "x")
        _write(workdir, "keep.py", "x")
        assert self._paths(backend, "**/*.py", workdir) == {"keep.py"}

    def test_a_pattern_that_names_a_hidden_entry_finds_it(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        _write(workdir, ".env", "x")
        assert self._paths(backend, ".env", workdir) == {".env"}

    def test_hidden_directory_named_in_the_pattern_is_searched(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        os.mkdir(os.path.join(workdir, ".github"))
        _write(workdir, ".github/ci.yml", "x")
        assert self._paths(backend, ".github/*.yml", workdir) == {".github/ci.yml"}

    def test_the_search_root_is_not_its_own_match(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        """`-name '*'` matches `.`, whose `%P` renders as an empty path."""
        _write(workdir, "a.txt", "x")
        assert "" not in self._paths(backend, "**/*", workdir)


@requires_gnu_userland
class TestGlobBoundaries:
    def test_trailing_space_in_a_matched_name_survives(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        _write(workdir, "z trailing ", "x")
        matches = backend.glob("*", workdir).matches
        assert {m["path"] for m in matches or []} == {"z trailing "}

    def test_symlink_escaping_the_root_is_dropped(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        """`BaseSandbox.glob` resolves each candidate and drops escapees;
        without that, rejecting `..` no longer keeps results inside `path`."""
        with tempfile.TemporaryDirectory() as outside:
            Path(outside, "secret.txt").write_text("x")
            os.symlink(outside, os.path.join(workdir, "escape"))
            _write(workdir, "inside.txt", "x")
            matches = backend.glob("**/*", workdir).matches
            assert {m["path"] for m in matches or []} == {"inside.txt"}


class TestGlobTranslationMatchesBaseSandbox:
    """Differential test: our `find` translation must select what upstream does.

    `BaseSandbox.glob` is the spec, and it is executable — it runs a `python3`
    script through `execute()`, which `LocalShellBackend` serves locally. So it
    can be used as an oracle directly, rather than restating its behaviour in
    assertions that can drift from it.

    Raw `glob.glob` is *not* the oracle: it reports the search root itself for
    `**`, which upstream filters out. Using it here produced a false failure.

    Running the real `glob()` needs GNU `find`, so those cases skip on macOS.
    Evaluating the generated primaries in-process needs nothing, so the
    semantics the protocol pins are checked on every platform. Hidden-file
    handling is why this exists: it was wrong twice, first by not excluding
    dotfiles at all, then by letting a named hidden segment re-open every
    segment below it.
    """

    TREE = [
        "visible/ci.yml",
        "visible/mod.py",
        ".hidden",
        ".git/config.py",
        ".github/ci.yml",
        ".github/public/ci.yml",
        ".github/.private/secret.yml",
        ".github/sub/.keep",
        ".github/.private/.keep",
        "top.py",
        ".env",
        ".gitignore",
        "cfg/.gitignore",
        ".config/sub/.gitignore",
        "sub/.env",
    ]

    PATTERNS = [
        "*",
        "*.py",
        "**/*.py",
        "**/*.yml",
        "**",
        ".env",
        ".github/*.yml",
        ".github/**/*.yml",
        "visible/*.py",
        # A hidden *final* segment must survive while the `**` before it still
        # refuses hidden directories (round-6 regression: `.config/sub/` leaked).
        "**/.gitignore",
        # A wildcard at a pinned position must not open hidden components even
        # when a later literal names a hidden entry.
        "*/.env",
        # A named hidden head plus a hidden literal tail across a `**` span.
        ".github/**/.keep",
    ]

    def _matches_primary(self, primary: str, path: str) -> bool:
        """Evaluate one `find` primary string against a `./`-prefixed path."""
        tokens = shlex.split(primary)
        depth = path.count("/")
        negate = False
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "!":
                negate = True
                i += 1
                continue
            value = tokens[i + 1]
            if token == "-mindepth":
                ok = depth >= int(value)
            elif token == "-maxdepth":
                ok = depth <= int(value)
            elif token == "-name":
                ok = fnmatch.fnmatch(path.rsplit("/", 1)[-1], value)
            elif token == "-path":
                ok = fnmatch.fnmatch(path, value)
            else:  # pragma: no cover - the vocabulary is closed
                raise AssertionError(f"unhandled primary token {token!r}")
            if negate:
                ok = not ok
                negate = False
            if not ok:
                return False
            i += 2
        return True

    @pytest.mark.parametrize("pattern", PATTERNS)
    def test_selection_matches_the_base_implementation(
        self, pattern: str, backend: LocalShellBackend, workdir: str
    ) -> None:
        for rel in self.TREE:
            entry = Path(workdir, rel)
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.write_text("x")

        every = sorted(
            os.path.relpath(str(p), workdir) for p in Path(workdir).rglob("*")
        )
        primaries = _shell._find_primaries(pattern)
        ours = sorted(
            rel
            for rel in every
            if any(self._matches_primary(p, "./" + rel) for p in primaries)
        )

        expected_result = BaseSandbox.glob(backend, pattern, workdir)
        assert expected_result.error is None
        expected = sorted(m["path"] for m in expected_result.matches or [])

        assert ours == expected, f"pattern {pattern!r}"


@requires_gnu_userland
class TestGlobRootBoundaries:
    """The containment check has to survive `glob()`'s documented default root."""

    def test_default_root_still_returns_matches(
        self, backend: LocalShellBackend
    ) -> None:
        """With `path=None` the root is `/`, so a naive `"$root"/*` prefix
        becomes `//*` and silently drops every candidate."""
        result = backend.glob("tmp")  # one level, no recursive walk of /
        assert result.error is None
        assert {m["path"] for m in result.matches or []} == {"tmp"}

    def test_a_sibling_sharing_the_root_prefix_is_still_excluded(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        """Normalizing the prefix must not let `/mnt/database` pass for
        `/mnt/data`; the separator is what makes the test a path test."""
        root = os.path.join(workdir, "data")
        os.mkdir(root)
        os.mkdir(os.path.join(workdir, "database"))
        Path(workdir, "database", "outside.txt").write_text("x")
        Path(root, "inside.txt").write_text("x")
        os.symlink(os.path.join(workdir, "database"), os.path.join(root, "sibling"))
        matches = backend.glob("**/*", root).matches
        assert {m["path"] for m in matches or []} == {"inside.txt"}


@requires_gnu_userland
class TestGlobHiddenSegments:
    """Naming a hidden directory must not re-open hidden entries below it."""

    def _paths(self, backend: LocalShellBackend, pattern: str, root: str) -> set[str]:
        return {m["path"] for m in backend.glob(pattern, root).matches or []}

    def _tree(self, workdir: str) -> None:
        for rel in (".github/public/ci.yml", ".github/.private/secret.yml"):
            entry = Path(workdir, rel)
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.write_text("needle\n")

    def test_named_hidden_dir_does_not_license_hidden_descendants(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        self._tree(workdir)
        assert self._paths(backend, ".github/**/*.yml", workdir) == {
            "public/ci.yml".replace("public", ".github/public")
        }

    def test_the_grep_slash_glob_route_agrees(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        """`grep` reuses the same translation, so it inherits the rule."""
        self._tree(workdir)
        result = backend.grep("needle", workdir, ".github/**/*.yml")
        assert result.error is None
        assert {m["path"] for m in result.matches or []} == {
            os.path.join(workdir, ".github/public/ci.yml")
        }


@requires_gnu_userland
class TestGlobPatternForms:
    """Pattern shapes Python `glob` accepts that the translation must honor."""

    def _paths(self, backend: LocalShellBackend, pattern: str, root: str) -> set[str]:
        result = backend.glob(pattern, root)
        assert result.error is None, result.error
        return {m["path"] for m in result.matches or []}

    def test_dot_segments_match_but_report_canonically(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        """`./*.txt` used to silently return nothing: the `.` segment inflated
        the depth pinning. It matches now; the `./` spelling is dropped from
        results (a documented divergence from Python's `./a.txt` form)."""
        Path(workdir, "a.txt").write_text("x")
        Path(workdir, ".hidden.txt").write_text("x")
        assert self._paths(backend, "./*.txt", workdir) == {"a.txt"}

    def test_interior_dot_segment(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        entry = Path(workdir, "src", "util", "helpers.py")
        entry.parent.mkdir(parents=True)
        entry.write_text("x")
        assert self._paths(backend, "src/./util/*.py", workdir) == {
            "src/util/helpers.py"
        }

    def test_trailing_slash_matches_directories_only(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        """As in Python: `*/` returns directories, spelled with the slash, and
        a symlink to a directory counts."""
        os.mkdir(os.path.join(workdir, "sub"))
        Path(workdir, "top.txt").write_text("x")
        os.symlink(os.path.join(workdir, "sub"), os.path.join(workdir, "dirlink"))
        assert self._paths(backend, "*/", workdir) == {"sub/", "dirlink/"}

    def test_recursive_trailing_slash_matches_the_base(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        for rel in ("a/b/f.txt", "c/g.txt"):
            entry = Path(workdir, rel)
            entry.parent.mkdir(parents=True)
            entry.write_text("x")
        expected = BaseSandbox.glob(backend, "**/", workdir)
        assert expected.error is None
        assert self._paths(backend, "**/", workdir) == {
            m["path"] for m in expected.matches or []
        }

    def test_head_doublestar_zero_expansion_matches_the_base(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        """`docs/**` includes `docs/` itself -- Python's zero-expansion form."""
        entry = Path(workdir, "docs", "deep", "file.py")
        entry.parent.mkdir(parents=True)
        entry.write_text("x")
        expected = BaseSandbox.glob(backend, "docs/**", workdir)
        assert expected.error is None
        base_paths = {m["path"] for m in expected.matches or []}
        assert "docs/" in base_paths
        assert self._paths(backend, "docs/**", workdir) == base_paths


@requires_gnu_userland
class TestGlobEmission:
    """The result records themselves, independent of pattern translation."""

    def test_backslash_filenames_survive_posix_echo(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        r"""Emitted with `printf`, never `echo`: under dash/ash, `echo` expands
        backslash escapes, so `with\ttab.txt` came back with a literal tab and
        an `end\c.txt` swallowed the record separator (round-6 regression)."""
        names = ["with\\ttab.txt", "with\\nnewline.txt", "end\\c.txt"]
        for name in names:
            Path(workdir, name).write_text("x")
        assert {m["path"] for m in backend.glob("*", workdir).matches or []} == set(
            names
        )

    def test_dangling_symlink_is_dropped(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        """`BaseSandbox.glob` stats every candidate, so a dangling symlink
        raises there and is dropped; `[ -e ]` mirrors that."""
        Path(workdir, "real.txt").write_text("x")
        os.symlink(os.path.join(workdir, "missing"), os.path.join(workdir, "dangle"))
        assert {m["path"] for m in backend.glob("*", workdir).matches or []} == {
            "real.txt"
        }


@requires_gnu_userland
class TestGlobRootProbes:
    """The root failures report the same codes `ls` uses."""

    def test_missing_root(self, backend: LocalShellBackend, workdir: str) -> None:
        result = backend.glob("*", os.path.join(workdir, "nope"))
        assert result.matches is None
        assert result.error is not None and "path_not_found" in result.error

    def test_file_root(self, backend: LocalShellBackend, workdir: str) -> None:
        target = Path(workdir, "plain.txt")
        target.write_text("x")
        result = backend.glob("*", str(target))
        assert result.matches is None
        assert result.error is not None and "not_a_directory" in result.error

    def test_unreadable_root(self, backend: LocalShellBackend, workdir: str) -> None:
        locked = os.path.join(workdir, "locked")
        os.mkdir(locked)
        os.chmod(locked, 0o000)
        try:
            result = backend.glob("*", locked)
        finally:
            os.chmod(locked, 0o755)
        if os.geteuid() == 0:  # pragma: no cover - root ignores modes
            pytest.skip("permission probes are meaningless as root")
        assert result.matches is None
        assert result.error is not None and "permission_denied" in result.error


@requires_gnu_userland
class TestReadNonRegularFiles:
    def test_fifo_reports_not_a_file_instead_of_hanging(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        """Without the `-f` probe, awk blocks on the FIFO until the command
        timeout; `BaseSandbox` reports `not_a_file` for the same case."""
        fifo = os.path.join(workdir, "pipe")
        os.mkfifo(fifo)
        result = backend.read(fifo)
        assert result.error is not None and "not_a_file" in result.error

    def test_relative_path_shaped_like_an_awk_assignment(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        """`awk prog V=1.txt` parses the operand as a variable assignment, so
        relative paths gain a `./` prefix before reaching awk."""
        Path(workdir, "V=1.txt").write_text("line1\n")
        program = _shell.build_read_command("V=1.txt", start=1, end=10)
        completed = subprocess.run(  # noqa: S603
            ["/bin/sh", "-c", program],
            capture_output=True,
            text=True,
            cwd=workdir,
            timeout=10,
        )
        chunk = completed.stdout.split("__SEP__\n")[2]
        assert base64.b64decode(chunk.strip()).decode() == "line1\n"


@requires_gnu_userland
class TestEditNonUtf8Content:
    def test_occurrence_count_survives_non_utf8_bytes(
        self, backend: LocalShellBackend, workdir: str
    ) -> None:
        """`LC_ALL=C` keeps gawk from writing a locale warning to stderr,
        which is the occurrence-count channel; without it the count parsed as
        `None` and a bogus lost-tail warning was logged."""
        target = Path(workdir, "bin.txt")
        target.write_bytes(b"\xff\xfe\x00header\nfoo\nfoo\n")
        result = backend.edit(str(target), "foo", "bar", replace_all=True)
        assert result.error is None
        assert result.occurrences == 2
        assert target.read_bytes() == b"\xff\xfe\x00header\nbar\nbar\n"


class TestReadUnderBothCaps:
    """`max_output_bytes` caps returned *content*, but `execute` also applies
    it to the wire, so a small cap once sliced the read's own base64 record
    and the window failed to decode. These run the real programs through the
    real `execute()` -- the harnesses that override `execute` cannot see it.
    """

    SERVICE_CAP = 4096

    def _post(self, _url: str, **kwargs: object) -> mock.Mock:
        """Run the command locally, capping each stream like the data plane."""
        command = kwargs["json"]["shellCommand"]  # type: ignore[index,call-overload]
        done = subprocess.run(  # noqa: S603
            ["/bin/sh", "-c", command], capture_output=True, text=True, timeout=30
        )
        response = mock.Mock(status_code=200, ok=True, content=b"")
        response.json.return_value = {
            "result": {
                "stdout": done.stdout.encode()[: self.SERVICE_CAP].decode(
                    "utf-8", "ignore"
                ),
                "stderr": done.stderr.encode()[: self.SERVICE_CAP].decode(
                    "utf-8", "ignore"
                ),
            },
            "status": str(done.returncode),
        }
        return response

    def _read(self, path: str, cap: int, **kwargs: int) -> ReadResult:
        backend = SessionsBashBackend(
            pool_management_endpoint="https://example.test/pool",
            access_token_provider=lambda: "token",
            max_output_bytes=cap,
        )
        with mock.patch("requests.post", side_effect=self._post):
            return backend.read(path, **kwargs)

    def test_window_within_a_small_cap_reads_intact(self, workdir: str) -> None:
        # 900 raw bytes are 1,200 of base64: over a 1,000-byte cap even though
        # the file itself is under it. Must cost round trips, not the content.
        path = _write(workdir, "line.txt", "z" * 900 + "\n")
        result = self._read(path, cap=1000)
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == "z" * 900

    def test_window_over_a_small_cap_truncates_to_whole_lines(
        self, workdir: str
    ) -> None:
        lines = [f"line {i:04d} {'y' * 60}" for i in range(60)]
        path = _write(workdir, "many.txt", "\n".join(lines) + "\n")
        result = self._read(path, cap=1000)
        assert result.error is None
        assert result.file_data is not None
        content = result.file_data["content"]
        assert content.endswith(TRUNCATION_MSG)
        shown = content[: -len(TRUNCATION_MSG)].split("\n")
        assert all(line in lines for line in shown if line)
        assert result.total_lines == 60
        assert result.next_offset == result.end_line
        # The resumed page starts exactly where this one stopped.
        assert result.next_offset is not None
        nxt = self._read(path, cap=1000, offset=result.next_offset)
        assert nxt.file_data is not None
        assert nxt.file_data["content"].split("\n")[0] == lines[result.next_offset]

    def test_cap_below_the_transport_floor_says_which_knob(self, workdir: str) -> None:
        # No chunk fits beside the bookkeeping; a smaller `limit` cannot help,
        # so the error names `max_output_bytes` rather than failing to decode.
        path = _write(workdir, "line.txt", "z" * 900 + "\n")
        result = self._read(path, cap=64)
        assert result.file_data is None
        assert result.error is not None
        assert "max_output_bytes" in result.error


class CappingShellBackend(LocalShellBackend):
    """A local shell whose output is cut at 4,096 bytes, like the data plane.

    The cut is silent and eats whatever falls past it -- including the
    completion marker -- which is exactly the live failure shape the chunked
    read exists for.
    """

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        result = super().execute(command, timeout=timeout)
        encoded = result.output.encode("utf-8")
        if len(encoded) > 4096:
            return ExecuteResponse(
                output=encoded[:4096].decode("utf-8", "ignore"),
                exit_code=result.exit_code,
                truncated=False,
            )
        return result


class TestChunkedReadThroughTheCap:
    """End to end against a real shell behind the 4 KiB cut: the chunked
    read must reassemble what a single-command read can no longer see."""

    def _backend(self) -> CappingShellBackend:
        return CappingShellBackend(
            pool_management_endpoint="https://example.test/pool",
            access_token_provider=lambda: "token",
        )

    def test_large_file_reads_fully(self, workdir: str) -> None:
        content = "".join(f"line {i:05d} {'x' * 30}\n" for i in range(400))  # ~17 KB
        path = _write(workdir, "big.txt", content)
        result = self._backend().read(path)
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == content[:-1]
        assert result.total_lines == 400
        assert result.next_offset is None

    def test_pagination_reassembles_across_chunk_boundaries(self, workdir: str) -> None:
        content = "".join(f"row {i:05d} {'y' * 40}\n" for i in range(600))
        path = _write(workdir, "paged.txt", content)
        backend = self._backend()
        first = backend.read(path, offset=0, limit=250)
        rest = backend.read(path, offset=first.next_offset or 0, limit=350)
        assert first.error is None and rest.error is None
        assert first.file_data is not None and rest.file_data is not None
        assert (
            first.file_data["content"] + "\n" + rest.file_data["content"]
            == content[:-1]
        )

    def test_multibyte_content_survives_chunk_boundaries(self, workdir: str) -> None:
        """Chunks cut at byte offsets; reassembly must not split characters."""
        content = ("é" * 50 + "\n") * 120  # 101 bytes/line, ~12 KB
        path = _write(workdir, "utf8.txt", content)
        result = self._backend().read(path)
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == content[:-1]
