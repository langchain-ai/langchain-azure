# 🦜️🔗 LangChain Azure

This repository contains the following packages with Azure integrations with LangChain:

- [langchain-azure-ai](https://pypi.org/project/langchain-azure-ai/)
- [langchain-azure-dynamic-sessions](https://pypi.org/project/langchain-azure-dynamic-sessions/)
- [langchain-sqlserver](https://pypi.org/project/langchain-sqlserver/)

**Note**: This repository will replace all Azure integrations currently present in the `langchain-community` package. Users are encouraged to migrate to this repository as soon as possible.

# Quick Start with langchain-azure-ai

The `langchain-azure-ai` package uses the [Azure AI Foundry SDK](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/develop/sdk-overview?tabs=sync&pivots=programming-language-python). This means you can use the package with a range of models including AzureOpenAI, Cohere, Llama, Phi-3/4, and DeepSeek-R1 to name a few. 

LangChain Azure AI also contains:
* [Azure AI Search](./libs/azure-ai/langchain_azure_ai/vectorstores)
* [Cosmos DB](./libs/azure-ai/langchain_azure_ai/vectorstores)
* [Azure AI Agent Service](./libs/azure-ai/langchain_azure_ai/azure_ai_agents)

Here's a quick start example to show you how to get started with the Chat Completions model. For more details and tutorials see [Develop with LangChain and LangGraph and models from Azure AI Foundry](https://aka.ms/azureai/langchain).

### Install langchain-azure

```bash
pip install -U langchain-azure-ai
```

### Azure AI Chat Completions Model with Azure OpenAI 

```python

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import HumanMessage, SystemMessage

model = AzureAIChatCompletionsModel(
    endpoint="https://{your-resource-name}.services.ai.azure.com/openai/v1",
    credential="your-api-key", #if using Entra ID you can should use DefaultAzureCredential() instead
    model_name="gpt-4o"
)

messages = [
    SystemMessage(
      content="Translate the following from English into Italian"
    ),
    HumanMessage(content="hi!"),
]

model.invoke(messages)
```

```python
AIMessage(content='Ciao!', additional_kwargs={}, response_metadata={'model': 'gpt-4o', 'token_usage': {'input_tokens': 20, 'output_tokens': 3, 'total_tokens': 23}, 'finish_reason': 'stop'}, id='run-0758e7ec-99cd-440b-bfa2-3a1078335133-0', usage_metadata={'input_tokens': 20, 'output_tokens': 3, 'total_tokens': 23})
```

### Azure AI Chat Completions Model with DeepSeek-R1 

```python

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.messages import HumanMessage, SystemMessage

model = AzureAIChatCompletionsModel(
    endpoint="https://{your-resource-name}.services.ai.azure.com/models",
    credential="your-api-key", #if using Entra ID you can should use DefaultAzureCredential() instead
    model_name="DeepSeek-R1",
)

messages = [
    HumanMessage(content="Translate the following from English into Italian: \"hi!\"")
]

message_stream = model.stream(messages)
print(' '.join(chunk.content for chunk in message_stream))
```

```python
 <think> 
 Okay ,  the  user  just  sent  " hi !"  and  I  need  to  translate  that  into  Italian .  Let  me  think .  " Hi "  is  an  informal  greeting ,  so  in  Italian ,  the  equivalent  would  be  " C iao !"  But  wait ,  there  are  other  options  too .  Sometimes  people  use  " Sal ve ,"  which  is  a  bit  more  neutral ,  but  " C iao "  is  more  common  in  casual  settings .  The  user  probably  wants  a  straightforward  translation ,  so  " C iao !"  is  the  safest  bet  here .  Let  me  double -check  to  make  sure  there 's  no  nuance  I 'm  missing .  N ope ,  " C iao "  is  definitely  the  right  choice  for  translating  " hi !"  in  an  informal  context .  I 'll  go  with  that . 
 </think> 

 C iao ! 
```

# Quick Start with Azure AI Agent Service 

### Basic Usage 

import agent service from langchain-azure-ai
```python
from langchain_azure_ai.azure_ai_agents import AzureAIAgentsService
```

```python
# Create an Azure AI Agents service using Azure AI Projects SDK
agent_service = AzureAIAgentsService(
    credential=DefaultAzureCredential(),
    endpoint=os.environ["PROJECT_ENDPOINT"],  # Use the project endpoint
    model="gpt-4.1",  # Use a model that's available in your project
    agent_name="langchain-demo-agent",
    instructions="You are a helpful AI assistant that provides clear and concise answers.",
)

# Test basic generation
response = agent_service.invoke("What is the capital of France?")
print(f"Response: {response}")
```    

# Welcome Contributors

Hi there! Thank you for even being interested in contributing to LangChain-Azure.
As an open-source project in a rapidly developing field, we are extremely open to contributions, whether they involve new features, improved infrastructure, better documentation, or bug fixes.


# Contribute Code

To contribute to this project, please follow the ["fork and pull request"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow.

Please follow the checked-in pull request template when opening pull requests. Note related issues and tag relevant
maintainers.

Pull requests cannot land without passing the formatting, linting, and testing checks first. See [Testing](#testing) and
[Formatting and Linting](#formatting-and-linting) for how to run these checks locally.

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

This repository contains three packages with Azure integrations with LangChain:
- [langchain-azure-ai](https://pypi.org/project/langchain-azure-ai/)
- [langchain-azure-dynamic-sessions](https://pypi.org/project/langchain-azure-dynamic-sessions/)
- [langchain-sqlserver](https://pypi.org/project/langchain-sqlserver/)

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
│   ├── langchain-sqlserver
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

