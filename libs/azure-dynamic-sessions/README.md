# langchain-azure-dynamic-sessions

This package contains the LangChain integration for Azure Container Apps dynamic sessions. You can use it to add a secure and scalable code interpreter to your agents.

## Installation

```bash
pip install -U langchain-azure-dynamic-sessions
```

## Configuration

You first need to create an Azure Container Apps session pool and obtain its management endpoint. Learn how to [configure a pool in Azure Container Apps](https://learn.microsoft.com/azure/container-apps/session-pool#configure-a-pool).

By default, both tools use `DefaultAzureCredential` to authenticate with Azure. If you're using a user-assigned managed identity, you must set the `AZURE_CLIENT_ID` environment variable to the ID of the managed identity. Only Microsoft Entra ID tokens from an identity belonging to the **Azure ContainerApps Session Executor** role on the session pool are authorized to call the pool management API.

```azurecli
az role assignment create \
    --role "Azure ContainerApps Session Executor" \
    --assignee <PRINCIPAL_ID> \
    --scope <SESSION_POOL_RESOURCE_ID>
```

## Usage

### SessionsPythonREPLTool

Use the `SessionsPythonREPLTool` tool to give your agent the ability to execute Python code.

```python
from langchain.agents import create_agent
from langchain_azure_dynamic_sessions.tools import SessionsPythonREPLTool


# get the management endpoint from the session pool in the Azure portal
tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)

agent = create_agent(model=llm, tools=[tool])
result = agent.invoke({"messages": [{"role": "user", "content": "What is the current time in Vancouver, Canada?"}]})
```

### SessionsBashTool

Use the `SessionsBashTool` tool to give your agent the ability to execute bash commands.

```python
from langchain.agents import create_agent
from langchain_azure_dynamic_sessions.tools import SessionsBashTool


# get the management endpoint from the session pool in the Azure portal
tool = SessionsBashTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)

agent = create_agent(model=llm, tools=[tool])
result = agent.invoke({"messages": [{"role": "user", "content": "List the files in the current directory."}]})
```

You can also execute bash commands directly:

```python
from langchain_azure_dynamic_sessions import SessionsBashTool


tool = SessionsBashTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)

# execute a bash command
result = tool.run("echo hello world")
# Returns: '{"stdout": "hello world\n", "stderr": "", "exitCode": 0}'
```

The tool supports file operations as well:

```python
tool = SessionsBashTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)

# upload a file to the session
tool.upload_file(local_file_path="./local_script.sh", remote_file_path="/mnt/user/script.sh")

# list files in the session
files = tool.list_files()

# download a file from the session
tool.download_file(remote_file_path="/mnt/user/output.txt", local_file_path="./output.txt")
```

