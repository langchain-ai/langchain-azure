"""Live contract tests for `SessionsBashBackend` against a Shell session pool.

The unit suites either stub `execute()` or run the generated program under the
local `/bin/sh`. Neither proves the program behaves the same way in the pool's
image, which is where `awk`, `find -printf` and `base64` actually have to
agree. The cases here are the ones that previously passed against mocks while
being wrong in a real shell.

Requires a **Shell**-typed session pool; a Python-typed pool will fail these.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

import pytest

# importorskip, not a plain import: `python-dotenv` and `deepagents` are only
# installed with the `test_integration` group and the extras, and a whole-tree
# run without them must skip at collection rather than error.
dotenv = pytest.importorskip("dotenv")
dotenv.load_dotenv()

pytest.importorskip("deepagents")

from langchain_azure_container_apps.dynamic_sessions._base import (  # noqa: E402
    _delete_session,
)
from langchain_azure_container_apps.dynamic_sessions.backends import (  # noqa: E402
    SessionsBashBackend,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

SHELL_POOL = os.getenv("AZURE_DYNAMIC_SESSIONS_SHELL_POOL_MANAGEMENT_ENDPOINT")

# Release CI runs `make integration_tests` with no Azure resources, so these
# must skip rather than fail when unconfigured.
pytestmark = pytest.mark.skipif(
    not SHELL_POOL,
    reason=(
        "Set AZURE_DYNAMIC_SESSIONS_SHELL_POOL_MANAGEMENT_ENDPOINT to a "
        "Shell-typed session pool to run these tests"
    ),
)


@pytest.fixture(scope="module")
def backend() -> Iterator[SessionsBashBackend]:
    """One session for the module, deleted afterwards.

    Explicit deletion rather than waiting out the pool's cooldown: leaked
    sessions hold quota on a shared pool. Deliberately not wrapped in a
    try/except -- a failing DELETE should fail loudly here, not leave state
    for the next run to trip over.
    """
    instance = SessionsBashBackend(
        pool_management_endpoint=SHELL_POOL,  # type: ignore[arg-type]
        session_id=f"itest-{uuid.uuid4()}",
    )
    yield instance
    _delete_session(
        pool_management_endpoint=SHELL_POOL,  # type: ignore[arg-type]
        session_id=instance.id,
        access_token=instance._access_token_provider(),  # noqa: SLF001
    )


@pytest.fixture
def workdir(backend: SessionsBashBackend) -> str:
    path = f"/tmp/itest-{uuid.uuid4().hex}"  # noqa: S108
    assert backend.execute(f"mkdir -p {path}").exit_code == 0
    return path


def _seed(backend: SessionsBashBackend, path: str, content: str) -> None:
    assert backend.write(path, content).path == path


class TestEditContract:
    def test_two_occurrences_on_one_line_are_ambiguous(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        path = f"{workdir}/f.txt"
        _seed(backend, path, "apple apple\n")
        result = backend.edit(path, "apple", "PEAR")
        assert result.error is not None
        assert "appears multiple times" in result.error
        assert backend.read(path).file_data["content"] == "apple apple"

    def test_replace_all_replaces_both_on_one_line(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        path = f"{workdir}/f.txt"
        _seed(backend, path, "apple apple\n")
        assert backend.edit(path, "apple", "PEAR", replace_all=True).occurrences == 2
        assert backend.read(path).file_data["content"] == "PEAR PEAR"

    def test_multiline_old_string(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        path = f"{workdir}/f.txt"
        _seed(backend, path, "Line 1\nLine 2\nLine 3\n")
        assert backend.edit(path, "Line 1\nLine 2", "REPLACED").occurrences == 1
        assert backend.read(path).file_data["content"] == "REPLACED\nLine 3"

    def test_backslashes_survive(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        path = f"{workdir}/f.txt"
        _seed(backend, path, "a\\nb\n")
        backend.edit(path, "\\n", "X")
        assert backend.read(path).file_data["content"] == "aXb"

    def test_replacement_may_contain_newlines(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        path = f"{workdir}/f.txt"
        _seed(backend, path, "abc\n")
        backend.edit(path, "b", "1\n2")
        assert backend.read(path).file_data["content"] == "a1\n2c"

    def test_unicode_round_trips(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        path = f"{workdir}/f.txt"
        _seed(backend, path, "héllo wörld ✓\n")
        backend.edit(path, "wörld", "münd")
        assert backend.read(path).file_data["content"] == "héllo münd ✓"

    def test_empty_old_string_is_rejected(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        path = f"{workdir}/f.txt"
        _seed(backend, path, "abc\n")
        assert "must not be empty" in str(backend.edit(path, "", "X").error)

    def test_lf_old_string_matches_a_crlf_file(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        path = f"{workdir}/crlf.txt"
        assert (
            backend.execute(f"printf 'alpha\\r\\nbeta\\r\\n' > {path}").exit_code == 0
        )
        content = backend.read(path).file_data["content"]
        assert content == "alpha\nbeta"
        assert backend.edit(path, content, "one\ntwo").error is None
        # Compare bytes through an exit status, not through captured output:
        # `execute()` is deliberately unprotected by the completion marker, and
        # a one-line assertion input inherits its 1.6-4% tail-loss rate. The
        # service returns the status in its own field, which cannot be lost.
        want = f"{path}.want"
        assert backend.execute(f"printf 'one\\r\\ntwo\\r\\n' > {want}").exit_code == 0
        assert backend.execute(f"cmp -s {path} {want}").exit_code == 0


class TestWriteContract:
    def test_refuses_to_overwrite(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        path = f"{workdir}/f.txt"
        _seed(backend, path, "original")
        assert "already exists" in str(backend.write(path, "replacement").error)
        assert backend.read(path).file_data["content"] == "original"

    def test_creates_parent_directories_with_spaces(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        path = f"{workdir}/a dir/nested/f.txt"
        assert backend.write(path, "x").path == path
        assert backend.read(path).file_data["content"] == "x"

    @pytest.mark.parametrize(
        "name", ["with space.txt", "with$(id).txt", "with`id`.txt", "with'quote.txt"]
    )
    def test_hostile_filenames_are_not_interpreted(
        self, backend: SessionsBashBackend, workdir: str, name: str
    ) -> None:
        path = f"{workdir}/{name}"
        assert backend.write(path, "payload").path == path
        assert backend.read(path).file_data["content"] == "payload"

    def test_shell_metacharacters_in_content_are_inert(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        path = f"{workdir}/f.txt"
        payload = "$(touch /tmp/pwned) `id` ${HOME}"  # noqa: S108
        _seed(backend, path, payload + "\n")
        assert backend.read(path).file_data["content"] == payload
        assert backend.execute("test -e /tmp/pwned").exit_code != 0


class TestListingContract:
    def test_ls_returns_absolute_paths(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        _seed(backend, f"{workdir}/f.txt", "x")
        assert backend.execute(f"mkdir -p {workdir}/sub").exit_code == 0
        entries = backend.ls(workdir).entries
        assert entries is not None
        assert {e["path"]: e["is_dir"] for e in entries} == {
            f"{workdir}/f.txt": False,
            f"{workdir}/sub": True,
        }

    def test_ls_missing_path_is_an_error(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        result = backend.ls(f"{workdir}/nope")
        assert result.entries is None
        assert "path_not_found" in str(result.error)

    def test_glob_matches_are_relative_to_the_root(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        _seed(backend, f"{workdir}/pkg/mod.py", "x")
        result = backend.glob("**/*.py", workdir)
        assert result.matches == [{"path": "pkg/mod.py", "is_dir": False}]

    def test_glob_missing_root_is_an_error_not_an_empty_result(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        result = backend.glob("*.py", f"{workdir}/nope")
        assert result.matches is None
        assert result.error is not None

    def test_listing_preserves_filename_whitespace(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        for name in (" leading.txt", "middle.txt", "z trailing "):
            _seed(backend, f"{workdir}/{name}", "x\n")
        entries = backend.ls(workdir).entries
        assert entries is not None
        assert sorted(e["path"].rsplit("/", 1)[-1] for e in entries) == [
            " leading.txt",
            "middle.txt",
            "z trailing ",
        ]

    def test_directory_symlink_is_not_reported_as_a_directory(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        assert backend.execute(f"mkdir -p {workdir}/realdir").exit_code == 0
        assert backend.execute(f"ln -s {workdir}/realdir {workdir}/link").exit_code == 0
        entries = backend.ls(workdir).entries
        assert entries is not None
        by_name = {e["path"].rsplit("/", 1)[-1]: e["is_dir"] for e in entries}
        assert by_name == {"realdir": True, "link": False}


class TestReadContract:
    def test_paginates(self, backend: SessionsBashBackend, workdir: str) -> None:
        path = f"{workdir}/f.txt"
        _seed(backend, path, "".join(f"L{i}\n" for i in range(10)))
        result = backend.read(path, offset=3, limit=2)
        assert result.file_data["content"] == "L3\nL4"
        assert result.total_lines == 10
        assert result.next_offset == 5

    @pytest.mark.parametrize("content", ["\n", "\n\n", "\na\n"])
    def test_blank_lines_produce_a_valid_window(
        self, backend: SessionsBashBackend, workdir: str, content: str
    ) -> None:
        path = f"{workdir}/blank.txt"
        _seed(backend, path, content)
        result = backend.read(path)
        assert result.error is None
        assert 1 <= result.start_line <= result.end_line

    def test_offset_past_end_is_an_error(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        path = f"{workdir}/f.txt"
        _seed(backend, path, "one\n")
        assert "exceeds file length" in str(backend.read(path, offset=99).error)

    def test_missing_file(self, backend: SessionsBashBackend, workdir: str) -> None:
        assert "file_not_found" in str(backend.read(f"{workdir}/nope").error)

    def test_window_beyond_the_service_output_cap_reads_fully(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        """The data plane cuts each execution's output at 4,096 bytes with no
        signal; the chunked read must page a ~12 KB window through it intact.
        Before chunking this returned 'the session returned no output' on
        every attempt."""
        content = "".join(f"line {i:04d} {'x' * 40}\n" for i in range(240))
        path = f"{workdir}/large.txt"
        _seed(backend, path, content)
        result = backend.read(path)
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == content[:-1]
        assert result.total_lines == 240
        assert result.next_offset is None


class TestExecuteContract:
    def test_output_at_the_service_cap_is_flagged_truncated(
        self, backend: SessionsBashBackend
    ) -> None:
        """`execute` is deliberately unmarked, so the cap would otherwise cut
        silently with exit 0; the at-cap flag is the only signal the caller
        gets."""
        result = backend.execute("printf '%8000s' '' | tr ' ' z")
        assert result.exit_code == 0
        assert result.truncated is True

    def test_small_output_is_not_flagged(self, backend: SessionsBashBackend) -> None:
        result = backend.execute("echo small")
        assert result.truncated is False


class TestTransferContract:
    def test_round_trips_through_the_flat_store(
        self, backend: SessionsBashBackend
    ) -> None:
        path = f"/mnt/data/itest-{uuid.uuid4().hex}.txt"
        uploaded = backend.upload_files([(path, b"payload")])
        assert [(r.path, r.error) for r in uploaded] == [(path, None)]
        downloaded = backend.download_files([path])
        assert [(r.path, r.content, r.error) for r in downloaded] == [
            (path, b"payload", None)
        ]

    def test_uploaded_file_is_readable_at_the_reported_path(
        self, backend: SessionsBashBackend
    ) -> None:
        # The old basename flattening reported a path the shell could not read.
        name = f"itest-{uuid.uuid4().hex}.txt"
        path = f"/mnt/data/{name}"
        assert backend.upload_files([(path, b"payload")])[0].error is None
        assert backend.read(path).file_data["content"] == "payload"

    @pytest.mark.parametrize(
        "path", ["/deep/nested/a.txt", "/mnt/data/sub/a.txt", "/mnt/database/a.txt"]
    )
    def test_unstorable_paths_are_rejected(
        self, backend: SessionsBashBackend, path: str
    ) -> None:
        assert backend.upload_files([(path, b"x")])[0].error == "invalid_path"
        assert backend.download_files([path])[0].error == "invalid_path"


class TestGrepContract:
    """`grep` is a new shell-native override; prove it against the pool image."""

    def test_matches_and_basename_glob(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        _seed(backend, f"{workdir}/app.py", "import os\nneedle here\n")
        _seed(backend, f"{workdir}/doc.txt", "needle too\n")
        result = backend.grep("needle", path=workdir, glob="*.py")
        assert result.error is None
        assert [(m["path"], m["line"], m["text"]) for m in result.matches or []] == [
            (f"{workdir}/app.py", 2, "needle here")
        ]

    def test_slash_glob_uses_the_find_route(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        assert backend.execute(f"mkdir -p {workdir}/src/sub").exit_code == 0
        _seed(backend, f"{workdir}/src/sub/lib.py", "needle deep\n")
        _seed(backend, f"{workdir}/top.py", "needle top\n")
        result = backend.grep("needle", path=workdir, glob="src/**/*.py")
        assert result.error is None
        assert [m["path"] for m in result.matches or []] == [
            f"{workdir}/src/sub/lib.py"
        ]

    def test_missing_root_is_reported(self, backend: SessionsBashBackend) -> None:
        result = backend.grep("needle", path="/tmp/itest-definitely-missing")
        assert result.matches is None
        assert result.error is not None
        assert "path_not_found" in result.error


class TestGlobContract:
    """Selection semantics against the pool image's own GNU `find`.

    The local-shell suite covers these too but skips without GNU userland, so
    for a macOS maintainer this is where they actually execute.
    """

    def test_wildcard_does_not_match_hidden_entries(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        _seed(backend, f"{workdir}/.hidden", "x\n")
        _seed(backend, f"{workdir}/visible", "x\n")
        matches = backend.glob("*", workdir).matches
        assert {m["path"] for m in matches or []} == {"visible"}

    def test_a_pattern_naming_a_hidden_entry_finds_it(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        _seed(backend, f"{workdir}/.env", "x\n")
        matches = backend.glob(".env", workdir).matches
        assert {m["path"] for m in matches or []} == {".env"}

    def test_recursive_wildcard_skips_hidden_directories(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        _seed(backend, f"{workdir}/.git/config.py", "x\n")
        _seed(backend, f"{workdir}/keep.py", "x\n")
        matches = backend.glob("**/*.py", workdir).matches
        assert {m["path"] for m in matches or []} == {"keep.py"}

    def test_glob_drops_a_symlink_escaping_the_root(
        self, backend: SessionsBashBackend, workdir: str
    ) -> None:
        outside = f"/tmp/outside-{uuid.uuid4().hex}"  # noqa: S108
        assert backend.execute(f"mkdir -p {outside}").exit_code == 0
        _seed(backend, f"{outside}/secret.txt", "x\n")
        _seed(backend, f"{workdir}/inside.txt", "x\n")
        assert backend.execute(f"ln -s {outside} {workdir}/escape").exit_code == 0
        matches = backend.glob("**/*", workdir).matches
        assert {m["path"] for m in matches or []} == {"inside.txt"}

    def test_the_default_root_still_returns_matches(
        self, backend: SessionsBashBackend
    ) -> None:
        """`glob()` defaults to `path=None`, i.e. a root of `/`. A naive
        `"$root"/*` containment prefix becomes `//*` and drops everything."""
        result = backend.glob("tmp")
        assert result.error is None
        assert {m["path"] for m in result.matches or []} == {"tmp"}
