"""Logic Apps tools."""

import json
from typing import Any, Dict, Optional

import requests
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr, model_validator

try:
    from azure.mgmt.logic import LogicManagementClient
except ImportError:
    raise ImportError(
        "To use the Azure Logic Apps tool, please install the 'azure-mgmt-logic'"
        "package: `pip install azure-mgmt-logic` or install the 'tools' extra: "
        "`pip install langchain-azure-ai[tools]`"
    )


class AzureLogicAppTool(BaseTool):
    """A tool that interacts with Azure Logic Apps."""

    name: str = "azure_logic_app_tool"
    """The name of the tool. Use a descriptive name that indicates its purpose."""

    description: str = (
        "A wrapper around Azure Logic Apps. "
        "Useful for when you need to invoke a Logic App. "
    )
    """A description of the tool that explains its functionality and usage.
    Use this description to help users understand when to use this tool."""

    subscription_id: str
    """Azure Subscription ID where the Logic Apps are hosted."""

    resource_group: str
    """Azure Resource Group where the Logic Apps are hosted."""

    credential: Optional[TokenCredential] = None
    """The API key or credential to use to connect to the service. I
    f None, DefaultAzureCredential is used."""

    logic_app_name: str
    """The name of the Logic App to invoke."""

    trigger_name: str
    """The name of the trigger in the Logic App to invoke."""

    _client: LogicManagementClient = PrivateAttr()

    _callback_url: str = PrivateAttr()

    @model_validator(mode="after")
    def initialize_client(self) -> "AzureLogicAppTool":
        """Initialize the Azure Logic Apps client."""
        credential: TokenCredential
        if self.credential is None:
            credential = DefaultAzureCredential()
        else:
            credential = self.credential

        self._client = LogicManagementClient(
            credential, self.subscription_id, user_agent="langchain-azure-ai"
        )

        self.register_logic_app(self.logic_app_name, self.trigger_name)

        return self

    def register_logic_app(self, logic_app_name: str, trigger_name: str) -> None:
        """Retrieves and stores a callback URL for a specific Logic App + trigger.

        Raises a ValueError if the callback URL is missing.
        """
        callback = self._client.workflow_triggers.list_callback_url(
            resource_group_name=self.resource_group,
            workflow_name=logic_app_name,
            trigger_name=trigger_name,
        )

        if callback.value is None:
            raise ValueError(
                f"No callback URL returned for Logic App '{logic_app_name}'."
            )

        self._callback_url = callback.value

    def invoke_logic_app(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invokes the registered Logic App (by name) with the given JSON payload.

        Returns a dictionary summarizing success/failure.
        """
        response = requests.post(url=self._callback_url, json=payload)

        if response.ok:
            return {"result": f"Successfully invoked {self.logic_app_name}."}
        else:
            return {
                "error": (
                    f"Error invoking {self.logic_app_name} "
                    f"({response.status_code}): {response.text}"
                )
            }

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            inputs = json.loads(query)
        except json.JSONDecodeError:
            inputs = {"input": query}

        response = self.invoke_logic_app(
            payload=inputs,
        )

        return json.dumps(response, indent=2)
