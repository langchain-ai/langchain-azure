"""Ask typed questions with Azure OpenAI Decisions using environment configuration."""

from langchain_azure_ai.decisions import (
    AzureAIOpenAIApiDecisions,
    Choice,
    Predicate,
    Score,
)


def main() -> None:
    """Print typed answers, including any refusals, for a customer request."""
    decisions = AzureAIOpenAIApiDecisions()
    response = decisions.invoke(
        {
            "input": "Checkout is broken. Please refund my payment.",
            "questions": {
                "refund": Predicate(instructions="Is a refund requested?"),
                "team": Choice(
                    instructions="Which team should handle this request?",
                    choices={
                        "billing": "Payment issues",
                        "engineering": "Product bugs",
                    },
                ),
                "severity": Score(
                    instructions="How severe is the problem?",
                    levels=["Minor", "Major"],
                ),
            },
        }
    )
    for name, answer in response.answers.items():
        print(name, answer.model_dump())
    print("Usage:", response.usage.model_dump())


if __name__ == "__main__":
    main()
