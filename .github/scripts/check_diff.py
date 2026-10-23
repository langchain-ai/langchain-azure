import json
import subprocess
import sys
from pathlib import Path

LIB_DIRS = [
    "libs/azure-dynamic-sessions",
    "libs/sqlserver",
    "libs/azure-ai",
    "libs/azure-postgresql",
    "libs/azure-storage",
    "libs/azure-cosmosdb",
    "libs/azure-compute",
]

REPO_ROOT = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).resolve().parent,
        encoding="utf-8",
    ).strip()
)
DOCS_DIRS = [
    directory
    for directory in [f"{lib}/docs" for lib in LIB_DIRS] + ["docs"]
    if (REPO_ROOT / directory).is_dir()
]

if __name__ == "__main__":
    files = sys.argv[1:]

    test_dirs: set[str] = set()
    example_dirs: set[str] = set()

    if len(files) >= 300:
        # max diff length is 300 files - there are likely files missing
        raise ValueError("Max diff reached. Please manually run CI on changed libs.")

    for file in files:
        if any(
            file.startswith(dir_)
            for dir_ in (
                ".github/workflows",
                ".github/tools",
                ".github/actions",
                ".github/scripts/check_diff.py",
            )
        ):
            test_dirs.update(LIB_DIRS)
            example_dirs.update([*DOCS_DIRS, "samples"])
        elif file.startswith(".github/tests/"):
            example_dirs.update([*DOCS_DIRS, "samples"])
        elif file.startswith("docs/"):
            example_dirs.add("docs")
        elif file.startswith("samples/"):
            parts = file.split("/")
            directory = "/".join(parts[:2]) if len(parts) > 2 else "samples"
            example_dirs.add(directory)
        elif file.startswith("libs/"):
            directory = "/".join(file.split("/")[:2])
            if directory not in LIB_DIRS:
                raise ValueError(
                    f"Unknown lib: {file}. check_diff.py likely needs an update "
                    "for this new library!"
                )
            docs_directory = f"{directory}/docs"
            if file.startswith(f"{docs_directory}/"):
                example_dirs.add(docs_directory)
            else:
                test_dirs.add(directory)
                example_dirs.add("samples")
                if docs_directory in DOCS_DIRS:
                    example_dirs.add(docs_directory)

    example_dirs = {
        directory for directory in example_dirs if (REPO_ROOT / directory).is_dir()
    }
    if "samples" in example_dirs:
        example_dirs = {
            directory
            for directory in example_dirs
            if not directory.startswith("samples/")
        }
    outputs = {
        "dirs-to-lint": sorted(test_dirs),
        "dirs-to-test": sorted(test_dirs),
        "dirs-to-test-docs-samples": sorted(example_dirs),
    }
    for key, value in outputs.items():
        json_output = json.dumps(value)
        print(f"{key}={json_output}")  # noqa: T201
