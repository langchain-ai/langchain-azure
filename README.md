# 🦜️🔗 LangChain Azure

This repository contains the following packages with Azure integrations with LangChain:

- [langchain-azure-ai](https://pypi.org/project/langchain-azure-ai/)
- [langchain-azure-dynamic-sessions](https://pypi.org/project/langchain-azure-dynamic-sessions/)
- [langchain-sqlserver](https://pypi.org/project/langchain-sqlserver/)
- [langchain-azure-postgresql](https://pypi.org/project/langchain-azure-postgresql/)
- [langchain-azure-storage](https://pypi.org/project/langchain-azure-storage/)

**Note**: This repository will replace all Azure integrations currently present in the `langchain-community` package. Users are encouraged to migrate to this repository as soon as possible.

# Quick Start with langchain-azure-ai

The `langchain-azure-ai` package uses the Microsoft Foundry family of SDKs and client libraries for Azure to provide first-class support of Microsoft Foundry capabilities in LangChain and LangGraph.

This package includes:

* [Microsoft Agent Service](./libs/azure-ai/langchain_azure_ai/agents)
* [Microsoft Foundry Models inference](./libs/azure-ai/langchain_azure_ai/chat_models)
* [Microsoft Foundry Content Safety](./libs/azure-ai/langchain_azure_ai/agents/middleware)
* [Microsoft Foundry Tools](./libs/azure-ai/langchain_azure_ai/tools)
* [Azure AI Search](./libs/azure-ai/langchain_azure_ai/vectorstores)
* [Azure AI Services tools](./libs/azure-ai/langchain_azure_ai/tools)
* [Cosmos DB](./libs/azure-ai/langchain_azure_ai/vectorstores)

Here's a quick start example to show you how to get started with the Chat Completions model. For more details and tutorials see [Develop with LangChain and LangGraph and models from Azure AI Foundry](https://aka.ms/azureai/langchain).

### Install langchain-azure

```bash
pip install -U langchain-azure-ai
```

### Microsoft Foundry Models

Use any Foundry Model with OpenAI-compatible APIs:

```python
from azure.identity import DefaultAzureCredential
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_core.messages import HumanMessage, SystemMessage

model = AzureAIOpenAIApiChatModel(
    project_endpoint="https://{your-resource-name}.services.ai.azure.com/api/projects/{your-project}",
    credential=DefaultAzureCredential(), # requires Azure AI Developer role. If using keys, use parameter `endpoint` instead of `project_endpoint`.
    model="gpt-5"                        # use any OpenAI-compatible model, like Mistral-Large-3
)

messages = [
    SystemMessage(
      content="Translate the following from English into Italian"
    ),
    HumanMessage(content="hi!"),
]

model.invoke(messages).pretty_print()
```

```output
================================== Ai Message ==================================
Ciao!
```

To use `init_chat_model` you must set the `AZURE_AI_PROJECT_ENDPOINT`, and (optional) `OPENAI_API_KEY` environment variables. Use the provider `azure_ai`:

```python 
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv 
load_dotenv()

# Option A) Using project endpoint
os.environ["AZURE_AI_PROJECT_ENDPOINT"] = "https://{your-resource-name}.services.ai.azure.com/api/projects/<project>"

# Option B) Using OpenAI endpoint
os.environ["OPENAI_API_BASE"] = "https://{your-resource-name}.services.ai.azure/openai/v1"
os.environ["OPENAI_API_KEY"] = "{your-key}"

model = init_chat_model("azure_ai:gpt-5-mini")
```

### Microsoft Foundry Agent Service

You can build multi agent graphs in LangGraph by using the integration with Microsoft Foundry Agent Service. The class `AgentServiceFactory` allows you to create agents and nodes that can be used to compose graphs.

```python
from azure.identity import DefaultAzureCredential
from langchain_core.messages import AIMessage, HumanMessage
from langchain_azure_ai.agents import AgentServiceFactory
from langchain_azure_ai.utils.agents import pretty_print

factory = AgentServiceFactory(
    project_endpoint="https://{your-resource-name}.services.ai.azure.com/api/projects/{your-project}",
    credential=DefaultAzureCredential()
)

echo_node = factory.get_agent_node(name="my-echo-agent", version="latest")
```

Agent Service nodes run in Microsoft Foundry but can be added to any graph:

```python
graph.add_node("expert_node", echo_node)
```

Use the graph as usual:

```python
agent = graph.compile()
messages = [HumanMessage(content="I'm a genius and I love programming!")]
response = agent.invoke({"messages": messages})

pretty_print(response)
```

```output
================================ Human Message =================================

I'm a genius and I love programming!
================================== Ai Message ==================================
Name: my-echo-agent

You're not a genius and you don't love programming!
```

You can also create a node manually to compose in your graph:

```python
from langchain_azure_ai.agents.prebuilt.tools import CodeInterpreterTool

coder_node = factory.create_declarative_chat_node(
    name="code-interpreter-agent",
    model="gpt-4.1",
    instructions="You are a helpful assistant that can run Python code.",
    tools=[CodeInterpreterTool()],
)

builder.add_node("coder", coder_node)
```

# Welcome Contributors

Hi there! Thank you for even being interested in contributing to LangChain-Azure.
As an open-source project in a rapidly developing field, we are extremely open to contributions, whether they involve new features, improved infrastructure, better documentation, or bug fixes.


# Contribute Code

To contribute to this project, please follow the ["fork and pull request"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow.

Please follow the checked-in pull request template when opening pull requests. Note related issues and tag relevant
maintainers.

Pull requests cannot land without passing the formatting, linting, and testing checks first. See [Git Hooks](#git-hooks) for how to set up local hooks that run these checks automatically, and [Testing](#testing) and
[Formatting and Linting](#formatting-and-linting) for how to run them manually.

It's essential that we maintain great documentation and testing. If you:
- Fix a bug
  - Add a relevant unit or integration test when possible. 
- Make an improvement
  - Update unit and integration tests when relevant.
- Add a feature
  - Add unit and integration tests.

If there's something you'd like to add or change, opening a pull request is the
best way to get our attention. Please tag one of our maintainers for review. 

## Dependency Management: Poetry and other env/dependency managers

This project utilizes [Poetry](https://python-poetry.org/) v1.7.1+ as a dependency manager.

❗Note: *Before installing Poetry*, if you use `Conda`, create and activate a new Conda env (e.g. `conda create -n langchain python=3.9`)

Install Poetry: **[documentation on how to install it](https://python-poetry.org/docs/#installation)**.

❗Note: If you use `Conda` or `Pyenv` as your environment/package manager, after installing Poetry,
tell Poetry to use the virtualenv python environment (`poetry config virtualenvs.prefer-active-python true`)

## Different packages

This repository contains four packages with Azure integrations with LangChain:
- [langchain-azure-ai](https://pypi.org/project/langchain-azure-ai/)
- [langchain-azure-dynamic-sessions](https://pypi.org/project/langchain-azure-dynamic-sessions/)
- [langchain-sqlserver](https://pypi.org/project/langchain-sqlserver/)
- [langchain-azure-storage](https://pypi.org/project/langchain-azure-storage/)

Each of these has its own development environment. Docs are run from the top-level makefile, but development
is split across separate test & release flows.

## Repository Structure

If you plan on contributing to LangChain-Google code or documentation, it can be useful
to understand the high level structure of the repository.

LangChain-Azure is organized as a [monorepo](https://en.wikipedia.org/wiki/Monorepo) that contains multiple packages.

Here's the structure visualized as a tree:

```text
.
├── libs
│   ├── azure-ai
│   ├── azure-dynamic-sessions
│   ├── azure-storage
│   ├── sqlserver
```

## Local Development Dependencies

Install development requirements (for running langchain, running examples, linting, formatting, tests, and coverage):

```bash
poetry install --with lint,typing,test,test_integration
```

Then verify dependency installation:

```bash
make test
```

If during installation you receive a `WheelFileValidationError` for `debugpy`, please make sure you are running
Poetry v1.6.1+. This bug was present in older versions of Poetry (e.g. 1.4.1) and has been resolved in newer releases.
If you are still seeing this bug on v1.6.1+, you may also try disabling "modern installation"
(`poetry config installer.modern-installation false`) and re-installing requirements.
See [this `debugpy` issue](https://github.com/microsoft/debugpy/issues/1246) for more details.

## Git Hooks

This repository ships pre-commit and pre-push hooks under `.githooks/` that automatically enforce the same
formatting and linting checks that CI requires. **Activate them once** after cloning so problems are caught
locally before you push:

```bash
git config core.hooksPath .githooks
```

| Hook | Triggered by | What it runs |
|------|-------------|--------------|
| `pre-commit` | `git commit` | `make format && make lint_package && make lint_tests` in `libs/azure-ai` |
| `pre-push` | `git push` | `make format && make lint_package && make lint_tests` for every `libs/` package whose files are included in the push |

> **Why are these required?**  Pull requests cannot land without passing formatting, linting, and type-checking.
> The hooks run the same checks locally so you can fix issues immediately instead of discovering them in CI.

### What to do when a hook fails

1. Read the output — `ruff` and `mypy` errors include the file, line, and a description.
2. Re-run formatting manually if needed: `cd libs/<package> && make format`.
3. Fix any remaining lint or type errors reported by `make lint_package` / `make lint_tests`.
4. Stage the fixed files (`git add`) and retry your `git commit` or `git push`.

If you need to bypass a hook in exceptional circumstances (e.g. a work-in-progress commit):

```bash
git push --no-verify    # skip pre-push
git commit --no-verify  # skip pre-commit
```

> ⚠️ Using `--no-verify` does not bypass CI — the same checks will run on your pull request.

## Code Formatting

Formatting for this project is done via [ruff](https://docs.astral.sh/ruff/rules/).

To run formatting for a library, run the same command from the relevant library directory:

```bash
cd libs/{LIBRARY}
make format
```

Additionally, you can run the formatter only on the files that have been modified in your current branch as compared to the master branch using the format_diff command:

```bash
make format_diff
```

This is especially useful when you have made changes to a subset of the project and want to ensure your changes are properly formatted without affecting the rest of the codebase.

## Linting

Linting for this project is done via a combination of [ruff](https://docs.astral.sh/ruff/rules/) and [mypy](http://mypy-lang.org/).

To run linting for docs, cookbook and templates:

```bash
make lint
```

To run linting for a library, run the same command from the relevant library directory:

```bash
cd libs/{LIBRARY}
make lint
```

In addition, you can run the linter only on the files that have been modified in your current branch as compared to the master branch using the lint_diff command:

```bash
make lint_diff
```

This can be very helpful when you've made changes to only certain parts of the project and want to ensure your changes meet the linting standards without having to check the entire codebase.

We recognize linting can be annoying - if you do not want to do it, please contact a project maintainer, and they can help you with it. We do not want this to be a blocker for good code getting contributed.

## Spellcheck

Spellchecking for this project is done via [codespell](https://github.com/codespell-project/codespell).
Note that `codespell` finds common typos, so it could have false-positive (correctly spelled but rarely used) and false-negatives (not finding misspelled) words.

To check spelling for this project:

```bash
make spell_check
```

To fix spelling in place:

```bash
make spell_fix
```

If codespell is incorrectly flagging a word, you can skip spellcheck for that word by adding it to the codespell config in the `pyproject.toml` file.

```python
[tool.codespell]
...
# Add here:
ignore-words-list =...
```

## Testing

All of our packages have unit tests and integration tests, and we favor unit tests over integration tests.

Unit tests run on every pull request, so they should be fast and reliable.

Integration tests run once a day, and they require more setup, so they should be reserved for confirming interface points with external services.

### Unit Tests

Unit tests cover modular logic that does not require calls to outside APIs.
If you add new logic, please add a unit test.
In unit tests we check pre/post processing and mocking all external dependencies.

To install dependencies for unit tests:

```bash
poetry install --with test
```

To run unit tests:

```bash
make test
```

To run unit tests in Docker:

```bash
make docker_tests
```

To run a specific test:

```bash
TEST_FILE=tests/unit_tests/test_imports.py make test
```

### Integration Tests

Integration tests cover logic that requires making calls to outside APIs (often integration with other services).
If you add support for a new external API, please add a new integration test.

**Warning:** Almost no tests should be integration tests.

  Tests that require making network connections make it difficult for other 
  developers to test the code.

  Instead favor relying on `responses` library and/or mock.patch to mock
  requests using small fixtures.

To install dependencies for integration tests:

```bash
poetry install --with test,test_integration
```

To run integration tests:

```bash
make integration_tests
```


For detailed information on how to contribute, see [LangChain contribution guide](https://python.langchain.com/docs/contributing/).

