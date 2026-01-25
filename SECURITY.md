# Security Policy

> **Last Updated**: 2026-01-24  
> **Version**: 1.0.0  
> **Classification**: Public

---

## Table of Contents

1. [Security Overview](#security-overview)
2. [Reporting Security Issues](#reporting-security-issues)
3. [Supported Versions](#supported-versions)
4. [Security Best Practices](#security-best-practices)
5. [Authentication & Authorization](#authentication--authorization)
6. [Data Protection](#data-protection)
7. [Secrets Management](#secrets-management)
8. [Network Security](#network-security)
9. [Compliance](#compliance)
10. [Security Checklist](#security-checklist)

---

## Security Overview

### Security Principles

This repository follows **Defense in Depth** security approach:

1. **Secrets Management** - No hardcoded credentials, environment variables only
2. **Least Privilege** - Minimal permissions for all components
3. **Input Validation** - Pydantic models for type safety
4. **Observability** - Comprehensive tracing for audit trails
5. **Authentication** - Support for Azure AD/Managed Identity

### Threat Model

| Threat | Risk Level | Mitigation Strategy |
|--------|------------|---------------------|
| Unauthorized API access | **High** | Azure AD authentication, API keys |
| Credential leakage | **High** | .env.example only, secrets in Azure Key Vault |
| Prompt injection | **Medium** | Input sanitization, LangSmith monitoring |
| Data exfiltration | **Medium** | No PII in logs, encrypted connections |
| Supply chain attack | **Low** | Dependency scanning, pinned versions |
| DoS attacks | **Low** | Rate limiting (implement as needed) |

---

## Reporting Security Issues

### Responsible Disclosure

We take security seriously. If you discover a security vulnerability:

**DO:**
- Report privately to: abhilashjaiswal0110@gmail.com
- Provide detailed description and reproduction steps
- Allow reasonable time for fix before public disclosure
- Work with maintainers to verify the fix

**DO NOT:**
- Publicly disclose the vulnerability before fix
- Exploit the vulnerability beyond verification
- Access or modify other users' data

### What to Report

Security issues include but are not limited to:
- Authentication bypass
- Privilege escalation
- Code injection vulnerabilities
- Exposed credentials or secrets
- Unsafe dependencies
- Data leakage

### Response Timeline

- **Initial Response**: Within 48 hours
- **Triage**: Within 7 days
- **Fix Development**: Based on severity
  - Critical: 1-3 days
  - High: 7-14 days
  - Medium: 30 days
  - Low: Next release cycle

---

## Supported Versions

| Version | Supported | Notes |
|---------|-----------|-------|
| 1.0.x   | ✅ Yes    | Current stable release |
| 0.1.x   | ⚠️ Limited | Security fixes only |
| < 0.1.0 | ❌ No     | End of life |

**Recommendation**: Always use the latest stable version (1.0.x)

---

## Security Best Practices

### 1. Environment Configuration

```bash
# ✅ DO: Use .env files (never commit)
cp .env.example .env
# Edit .env with real values

# ❌ DON'T: Hardcode secrets
AZURE_OPENAI_API_KEY = "sk-..." # WRONG!
```

### 2. Azure Managed Identity (Recommended)

```python
# ✅ DO: Use DefaultAzureCredential (no secrets needed)
from azure.identity import DefaultAzureCredential
from langchain_azure_ai.agents import AgentServiceFactory

factory = AgentServiceFactory(
    project_endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
    credential=DefaultAzureCredential()  # Uses managed identity
)

# ❌ DON'T: Use hardcoded API keys in code
factory = AgentServiceFactory(
    credential="your-api-key-here"  # WRONG!
)
```

### 3. Secrets in Azure Key Vault

```python
# ✅ DO: Retrieve secrets from Azure Key Vault
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
client = SecretClient(
    vault_url="https://your-vault.vault.azure.net/",
    credential=credential
)
api_key = client.get_secret("OPENAI-API-KEY").value
```

### 4. Input Validation

```python
# ✅ DO: Use Pydantic for validation
from pydantic import BaseModel, Field, validator

class AgentRequest(BaseModel):
    query: str = Field(..., max_length=1000)
    session_id: str = Field(..., regex=r"^[a-zA-Z0-9_-]+$")
    
    @validator('query')
    def sanitize_query(cls, v):
        # Remove potentially malicious content
        return v.strip()
```

### 5. Logging and Monitoring

```python
# ✅ DO: Sanitize PII in logs
import logging
logger = logging.getLogger(__name__)

def safe_log(message: str, **kwargs):
    # Remove sensitive data before logging
    sanitized = {k: v for k, v in kwargs.items() 
                 if k not in ['password', 'api_key', 'token']}
    logger.info(message, extra=sanitized)

# ❌ DON'T: Log sensitive data
logger.info(f"User API key: {api_key}")  # WRONG!
```

---

## Authentication & Authorization

### Azure AD Integration

This repository supports Azure AD (Entra ID) authentication:

```bash
# .env configuration
AUTH_ENABLED=true
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
```

### API Key Authentication

For simple deployments:

```bash
# Generate secure API key
API_KEY=$(openssl rand -hex 32)
API_KEY_ENABLED=true
```

### Role-Based Access Control (RBAC)

Implement RBAC for production:

```python
from enum import Enum

class Role(Enum):
    VIEWER = "viewer"      # Read-only access
    USER = "user"          # Standard operations
    OPERATOR = "operator"  # Elevated operations
    ADMIN = "admin"        # Full access

# Check permissions before operations
def require_role(required_role: Role):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if current_user.role.value < required_role.value:
                raise PermissionError("Insufficient permissions")
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

---

## Data Protection

### Data Classification

| Data Type | Classification | Storage | Retention |
|-----------|----------------|---------|-----------|
| API Keys | **Secret** | Azure Key Vault | Rotate every 90 days |
| User queries | **Internal** | LangSmith (encrypted) | 30 days |
| Agent responses | **Internal** | In-memory only | Session only |
| Telemetry | **Internal** | Application Insights | 90 days |
| Audit logs | **Internal** | Azure Storage | 1 year |

### Data in Transit

- ✅ **HTTPS Only** - All external communications must use TLS 1.2+
- ✅ **Azure Private Link** - Use for internal Azure service connections
- ✅ **Certificate Validation** - Always validate SSL certificates

### Data at Rest

- ✅ **Encryption** - Enable encryption at rest for all storage
- ✅ **Key Management** - Use Azure Key Vault for key management
- ✅ **No Local Storage** - Avoid persisting sensitive data locally

### PII Handling

```python
# Pattern for PII redaction
import re

def redact_pii(text: str) -> str:
    """Redact common PII patterns."""
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                  '[EMAIL]', text)
    # Phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 
                  '[PHONE]', text)
    # SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', 
                  '[SSN]', text)
    return text
```

---

## Secrets Management

### Azure Key Vault Integration

**Setup**:

```bash
# Create Key Vault
az keyvault create \
  --name your-keyvault \
  --resource-group your-rg \
  --location eastus

# Grant access to Managed Identity
az keyvault set-policy \
  --name your-keyvault \
  --object-id your-managed-identity-id \
  --secret-permissions get list
```

**Usage**:

```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Initialize client
credential = DefaultAzureCredential()
vault_url = f"https://your-keyvault.vault.azure.net/"
client = SecretClient(vault_url=vault_url, credential=credential)

# Retrieve secrets
openai_key = client.get_secret("AZURE-OPENAI-API-KEY").value
langsmith_key = client.get_secret("LANGSMITH-API-KEY").value
```

### Environment Variable Security

```bash
# ✅ DO: Use .env files locally (gitignored)
# ✅ DO: Use Azure App Configuration for cloud deployments
# ✅ DO: Use environment variables in containers

# ❌ DON'T: Commit .env files
# ❌ DON'T: Echo secrets in scripts
# ❌ DON'T: Pass secrets via command line
```

### Secret Rotation

**Best Practice Schedule**:
- **API Keys**: Every 90 days
- **Passwords**: Every 60 days
- **Certificates**: Before expiration (monitor at 30 days)

---

## Network Security

### CORS Configuration

```python
# Production CORS settings
CORS_ORIGINS = [
    "https://your-prod-domain.com",
    "https://your-staging-domain.com",
]

# ❌ NEVER use in production:
CORS_ORIGINS = ["*"]  # WRONG!
```

### Firewall Rules

**Azure AI Foundry**:
- Restrict to specific IP ranges
- Use Azure Private Link for internal traffic
- Enable Azure Firewall for egress control

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/agent/invoke")
@limiter.limit("10/minute")  # 10 requests per minute
async def invoke_agent(request: Request):
    pass
```

---

## Compliance

### Regulatory Requirements

| Compliance Framework | Status | Notes |
|---------------------|--------|-------|
| **GDPR** | ✅ Compliant | No PII storage, data portability |
| **SOC 2** | 🔄 In Progress | Azure infrastructure certified |
| **HIPAA** | ⚠️ Not Certified | Do not use for health data |
| **ISO 27001** | ✅ Compliant | Via Azure certification |

### Data Residency

- **Azure Region**: Configurable per deployment
- **Data Sovereignty**: Respects regional requirements
- **Cross-Border**: Can be restricted via Azure policies

### Audit Logging

Enable comprehensive audit logs:

```bash
# Application Insights
APPLICATIONINSIGHTS_CONNECTION_STRING=your-connection-string

# LangSmith for agent traces
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
```

---

## Security Checklist

### Development Phase

- [ ] No hardcoded secrets in code
- [ ] `.env` files in `.gitignore`
- [ ] Input validation with Pydantic
- [ ] Error messages don't leak sensitive info
- [ ] Dependency scanning enabled
- [ ] Code review includes security check

### Pre-Deployment Phase

- [ ] `.env.example` provided (no real secrets)
- [ ] Secrets moved to Azure Key Vault
- [ ] HTTPS/TLS configured
- [ ] CORS properly configured (no wildcards)
- [ ] API authentication enabled
- [ ] Rate limiting implemented

### Production Phase

- [ ] Azure Managed Identity configured
- [ ] Application Insights enabled
- [ ] LangSmith tracing enabled
- [ ] Monitoring alerts configured
- [ ] Backup and recovery tested
- [ ] Incident response plan documented
- [ ] Security audit completed
- [ ] Penetration testing performed

### Ongoing Maintenance

- [ ] Rotate secrets every 90 days
- [ ] Review access logs monthly
- [ ] Update dependencies monthly
- [ ] Security patches applied within 7 days
- [ ] Incident response drills quarterly
- [ ] Security training for team annually

---

## Security Tools and Resources

### Recommended Tools

1. **Dependency Scanning**:
   ```bash
   pip install safety
   safety check
   ```

2. **Secret Scanning**:
   ```bash
   pip install detect-secrets
   detect-secrets scan
   ```

3. **Code Analysis**:
   ```bash
   pip install bandit
   bandit -r libs/
   ```

4. **Azure Security**:
   - Azure Security Center
   - Azure Sentinel
   - Microsoft Defender for Cloud

### Additional Resources

- [Azure Security Best Practices](https://docs.microsoft.com/azure/security/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [LangChain Security Guide](https://docs.langchain.com/docs/security)
- [Azure AI Foundry Security](https://aka.ms/azureai/security)

---

## Contact

**Security Team**: abhilashjaiswal0110@gmail.com  
**Response Time**: 48 hours for security issues  
**PGP Key**: Available upon request

---

**Document Version**: 1.0.0  
**Last Review**: 2026-01-24  
**Next Review**: 2026-04-24 (Quarterly)
