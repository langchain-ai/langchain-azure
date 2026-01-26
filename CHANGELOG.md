# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enterprise repository structure with comprehensive documentation
- Production readiness certification (Software, Security, Data Architecture)
- Comprehensive `.env.example` with all Azure AI Foundry parameters
- Security documentation (SECURITY.md) with threat model and mitigation strategies
- Contributing guidelines (CONTRIBUTING.md) with development workflow
- Code of Conduct (CODE_OF_CONDUCT.md) following Contributor Covenant
- Citation file (CITATION.cff) for academic references
- EditorConfig for consistent code formatting across IDEs
- CODEOWNERS file for code review assignments
- Comprehensive .gitignore and .dockerignore files
- Phase 1 setup documentation (PHASE1_PROGRESS.md, PHASE1_QUICKSTART.md)
- Enterprise knowledge base (Knowledge.md)
- Agent development guide (Agents.md)
- Deployment planning documentation (DEPLOYMENT_PLAN.md)

### Changed
- Enhanced README.md with enterprise standards alignment
- Improved .env.example with detailed Azure AI Foundry configuration
- Updated .gitignore with comprehensive exclusion patterns

### Security
- Implemented defense-in-depth security principles
- Added secrets management best practices
- Documented authentication and authorization patterns
- Established security certification process

## [1.0.4] - 2024-XX-XX

### Fixed
- Issue with dependencies resolution for `azure-ai-agents`
- `AzureAIOpenTelemetryTracer` span context propagation
- Context deallocation in OpenTelemetry tracer
- Environment variables `AZURE_AI_*` reading order

### Improved
- `AzureAIOpenTelemetryTracer` test coverage
- Internal tracing implementation

## [1.0.2] - 2024-XX-XX

### Changed
- `AzureAIOpenTelemetryTracer` now creates parent trace for multi-agent scenarios automatically

## [1.0.0] - 2024-XX-XX

### Added
- Support for LangChain and LangGraph 1.0

## [0.1.8] - 2024-XX-XX

### Fixed
- Multiple issues with `AzureAIOpenTelemetryTracer` (hierarchy, tool spans, finish reason, conversation ID)
- Image input handling for declarative agents in Azure AI Foundry

### Enhanced
- Tool descriptions for improved tool call accuracy

## [0.1.7] - 2024-XX-XX

### Added
- **[NEW]** LangGraph support for declarative agents created in Azure AI Foundry
- `AgentServiceFactory` for composing complex graphs

### Fixed
- Interface issue with `AzureAIEmbeddingsModel`
- Tool signatures for `AzureAIDocumentIntelligenceTool`, `AzureAIImageAnalysisTool`, and `AzureAITextAnalyticsHealthTool`

## [0.1.6] - 2024-XX-XX

### Changed
- **[Breaking]** Removed parameter `project_connection_string` (use `project_endpoint` instead)
- **[Breaking]** Removed class `AzureAIInferenceTracer` (use `AzureAIOpenTelemetryTracer`)

### Added
- Azure AI Services tools: `AzureAIDocumentIntelligenceTool`, `AzureAIImageAnalysisTool`, `AzureAITextAnalyticsHealthTool`
- `AIServicesToolkit` for unified access to Azure AI Services

## [0.1.4] - 2024-XX-XX

### Fixed
- Bug fixes from community feedback

## [0.1.3] - 2024-XX-XX

### Changed
- **[Breaking]** Renamed parameter `model_name` to `model` in `AzureAIEmbeddingsModel` and `AzureAIChatCompletionsModel`

### Fixed
- JSON mode issues in chat models
- NumPy dependencies compatibility
- Pydantic object tracing in inputs
- Made `connection_string` parameter optional

## [0.1.2] - 2024-XX-XX

### Fixed
- Community-reported bug fixes

## [0.1.1] - 2024-XX-XX

### Added
- `AzureCosmosDBNoSqlVectorSearch` and `AzureCosmosDBNoSqlSemanticCache`
- `AzureCosmosDBMongoVCoreVectorSearch` and `AzureCosmosDBMongoVCoreSemanticCache`
- `project_connection_string` parameter for direct AI project connection
- Native LLM structured outputs support (`json_schema` and `json_mode`)

### Fixed
- Multiple community-reported issues

## [0.1.0] - 2024-XX-XX

### Added
- Initial release
- `AzureAIEmbeddingsModel` for embedding generation
- `AzureAIChatCompletionsModel` for chat completions
- Azure AI Inference API support
- GitHub Models endpoint support
- `AzureAIOpenTelemetryTracer` for OpenTelemetry tracing

---

## Release Notes Guidelines

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (x.0.0): Breaking changes
- **MINOR** (0.x.0): New features, backward compatible
- **PATCH** (0.0.x): Bug fixes, backward compatible

### Change Categories

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security improvements

### Links

[Unreleased]: https://github.com/abhilashjaiswal0110/langchain-azure/compare/v1.0.4...HEAD
[1.0.4]: https://github.com/abhilashjaiswal0110/langchain-azure/compare/v1.0.2...v1.0.4
[1.0.2]: https://github.com/abhilashjaiswal0110/langchain-azure/compare/v1.0.0...v1.0.2
[1.0.0]: https://github.com/abhilashjaiswal0110/langchain-azure/compare/v0.1.8...v1.0.0
[0.1.8]: https://github.com/abhilashjaiswal0110/langchain-azure/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/abhilashjaiswal0110/langchain-azure/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/abhilashjaiswal0110/langchain-azure/compare/v0.1.4...v0.1.6
[0.1.4]: https://github.com/abhilashjaiswal0110/langchain-azure/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/abhilashjaiswal0110/langchain-azure/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/abhilashjaiswal0110/langchain-azure/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/abhilashjaiswal0110/langchain-azure/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/abhilashjaiswal0110/langchain-azure/releases/tag/v0.1.0
