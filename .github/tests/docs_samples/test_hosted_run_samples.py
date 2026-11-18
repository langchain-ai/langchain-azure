"""Keep azd and Docker startup consistent for graph-export samples."""

import json
import shlex
from pathlib import Path

import pytest
import yaml

SAMPLES = (
    Path(__file__).resolve().parents[3] / "samples/hosting/langgraph-hosted-agents"
)


@pytest.mark.parametrize("relative", ["responses/09_run", "invocations/03_run"])
def test_azd_starts_graph_host(relative: str, docs_samples_dirs: list[Path]) -> None:
    sample = SAMPLES / relative
    if not any(sample.is_relative_to(directory) for directory in docs_samples_dirs):
        pytest.skip("This check covers the graph-export hosting samples.")

    config = yaml.safe_load((sample / "azure.yaml").read_text(encoding="utf-8"))
    (service,) = config["services"].values()
    command = shlex.split(service["startupCommand"])
    assert command == [
        "python",
        "-m",
        "langchain_azure_ai.agents.hosting.run",
        "--protocol",
        relative.split("/")[0],
    ]
    dockerfile = (sample / "Dockerfile").read_text(encoding="utf-8")
    docker_command = next(
        line[4:] for line in dockerfile.splitlines() if line.startswith("CMD ")
    )
    assert command == json.loads(docker_command)
