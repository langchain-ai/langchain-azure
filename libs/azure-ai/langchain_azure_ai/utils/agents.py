"""Utility functions for agents."""

import base64

from langchain_core.messages import AIMessage


def pretty_print(response: dict, messages_key: str = "messages") -> None:
    """Pretty print the response from an agent execution.

    Args:
        response (dict): The response dictionary from an agent execution.
        messages_key (str): The key in the response that contains the list of messages.
            Defaults to "messages".
    """
    if messages_key not in response:
        raise ValueError(f"Response does not contain '{messages_key}' key.")

    for m in response[messages_key]:
        m.pretty_print()
        if isinstance(m, AIMessage) and not isinstance(m.content, str):
            for block in m.content:
                if isinstance(block, dict) and block.get("type") == "image":
                    image_data = block.get("base64")
                    if image_data:
                        try:
                            from IPython.display import Image, display  # type: ignore

                            display(Image(data=base64.b64decode(image_data)))
                        except ImportError:
                            print("IPython.display is not available.")
    if "__interrupt__" in response:
        for interrupt in response["__interrupt__"]:
            print(
                "\n================================== Interrupt ======================="
                "==========="
            )
            print(f"Interrupt ID: {interrupt.id}")
            print("Interrupt Value:")
            for tool_call in interrupt.value:
                print(f"\tTool Call ID: {tool_call['id']}")
                print(f"\tServer Label: {tool_call['server_label']}")
                print(f"\tTool Name: {tool_call['tool_name']}")
                print(f"\tArguments: {tool_call['arguments']}")
