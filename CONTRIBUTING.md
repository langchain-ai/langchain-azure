# Contributing to LangChain Azure

Thank you for your interest in contributing to the LangChain Azure integration! This document provides guidelines for contributing to this project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
3. [Development Setup](#development-setup)
4. [Pull Request Process](#pull-request-process)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)

---

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

---

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected vs actual behavior**
- **Environment details** (Python version, OS, Azure region)
- **Code samples** (if applicable)
- **Error messages and stack traces**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide detailed description** of the proposed feature
- **Explain why this would be useful** to most users
- **Include code examples** showing how it would work

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our coding standards
3. **Add tests** for any new functionality
4. **Update documentation** as needed
5. **Ensure all tests pass**
6. **Submit a pull request**

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Azure subscription with:
  - Azure AI Foundry project
  - Azure OpenAI service
  - Application Insights (optional)
- Git

### Local Setup

```bash
# Clone the repository
git clone https://github.com/abhilashjaiswal0110/langchain-azure.git
cd langchain-azure

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e "libs/azure-ai[opentelemetry,tools,dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your Azure credentials

# Run tests
pytest libs/azure-ai/tests/unit_tests/
```

### Development Tools

We use these tools to maintain code quality:

- **Ruff**: Fast Python linter and formatter
- **MyPy**: Static type checker
- **Pytest**: Testing framework
- **Pre-commit**: Git hooks for code quality

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run linting
ruff check libs/azure-ai/

# Run type checking
mypy libs/azure-ai/langchain_azure_ai/

# Run tests
pytest libs/azure-ai/tests/
```

---

## Pull Request Process

### 1. Branch Naming

Use descriptive branch names:

- `feature/add-cosmos-db-support`
- `fix/agent-factory-timeout`
- `docs/update-readme-examples`
- `refactor/simplify-auth-flow`

### 2. Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(agents): add support for streaming responses

Implements server-sent events for real-time agent progress updates.
Includes tests and documentation.

Closes #123
```

### 3. Code Review Process

All submissions require review. We follow these guidelines:

1. **Automated checks must pass** (tests, linting, type checking)
2. **At least one approval** from maintainers
3. **Documentation updated** if needed
4. **Changelog updated** for user-facing changes
5. **No merge conflicts** with main branch

### 4. Merging

- **Squash and merge** for feature branches
- **Linear history** preferred
- **Delete branch** after merge

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with these specifics:

- **Line length**: 88 characters (Black/Ruff default)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Docstrings**: Google style

### Type Hints

Always use type hints for function signatures:

```python
from typing import List, Optional

def create_agent(
    name: str,
    model: str,
    tools: List[BaseTool],
    trace: bool = True
) -> CompiledStateGraph:
    """Create a new agent with specified configuration.
    
    Args:
        name: Agent identifier
        model: Model deployment name
        tools: List of tools available to agent
        trace: Enable OpenTelemetry tracing
        
    Returns:
        Compiled LangGraph state graph
        
    Raises:
        ValueError: If model deployment not found
    """
    pass
```

### Docstring Format

Use Google-style docstrings:

```python
def process_document(
    file_path: str,
    extract_tables: bool = False
) -> Dict[str, Any]:
    """Process a document and extract content.
    
    This function uses Azure AI Document Intelligence to process
    documents of various formats and extract text, tables, and metadata.
    
    Args:
        file_path: Path to the document file. Supported formats:
            PDF, DOCX, PPTX, images (PNG, JPEG, TIFF).
        extract_tables: Whether to extract tables from the document.
            Default is False for better performance.
            
    Returns:
        Dictionary containing:
            - text: Extracted text content
            - tables: List of tables (if extract_tables=True)
            - metadata: Document metadata (pages, language, etc.)
            
    Raises:
        FileNotFoundError: If file_path doesn't exist
        ValueError: If file format is unsupported
        
    Example:
        >>> result = process_document("contract.pdf", extract_tables=True)
        >>> print(result["text"])
        >>> for table in result["tables"]:
        ...     print(table)
    """
    pass
```

### Error Handling

```python
# ✅ DO: Specific exceptions with context
try:
    result = agent.invoke({"messages": messages})
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise
except ConnectionError as e:
    logger.error(f"Azure service unavailable: {e}")
    raise ServiceUnavailableError("Cannot connect to Azure AI")

# ❌ DON'T: Bare except or generic exceptions
try:
    result = agent.invoke({"messages": messages})
except:  # WRONG!
    pass
```

---

## Testing Guidelines

### Test Structure

```
libs/azure-ai/tests/
├── unit_tests/           # Fast, isolated tests
│   ├── test_agents.py
│   └── test_chat_models.py
└── integration_tests/    # Tests requiring Azure resources
    ├── test_agent_e2e.py
    └── conftest.py       # Shared fixtures
```

### Writing Tests

```python
import pytest
from langchain_azure_ai.agents import AgentServiceFactory

def test_agent_creation():
    """Test basic agent creation."""
    factory = AgentServiceFactory()
    agent = factory.create_prompt_agent(
        name="test-agent",
        model="gpt-4o",
        instructions="You are a test agent."
    )
    assert agent is not None
    assert hasattr(agent, "invoke")

@pytest.mark.integration
def test_agent_invocation(azure_credentials):
    """Test agent invocation with Azure services."""
    factory = AgentServiceFactory(**azure_credentials)
    agent = factory.create_prompt_agent(
        name="test-agent",
        model="gpt-4o",
        instructions="Reply with 'Hello World'"
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "Hi"}]})
    assert "Hello World" in result["messages"][-1].content
```

### Test Coverage

- **Aim for >80% coverage** for new code
- **Unit tests** for all public functions
- **Integration tests** for Azure service interactions
- **Mock Azure services** in unit tests

```bash
# Run with coverage report
pytest --cov=langchain_azure_ai --cov-report=html libs/azure-ai/tests/
```

---

## Documentation

### Code Documentation

- **All public APIs** must have docstrings
- **Complex logic** should have inline comments
- **Examples** in docstrings when helpful

### README Updates

Update README.md when:
- Adding new features
- Changing installation instructions
- Modifying configuration

### Changelog

Update `CHANGELOG.md` for all user-facing changes:

```markdown
## [Unreleased]

### Added
- Support for Azure Cosmos DB vector store (#123)
- Streaming responses for agents (#456)

### Changed
- Improved error messages for authentication failures (#789)

### Fixed
- Agent timeout issue with long-running tasks (#101)

### Deprecated
- `project_connection_string` parameter (use `project_endpoint`)
```

---

## Questions?

- **GitHub Discussions**: For general questions
- **GitHub Issues**: For bugs and feature requests
- **Email**: abhilashjaiswal0110@gmail.com

Thank you for contributing! 🎉
