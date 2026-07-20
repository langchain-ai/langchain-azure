from client import build_parser


def test_tokendelay_is_accepted() -> None:
    args = build_parser().parse_args(
        [
            "--endpoint",
            "https://example.services.ai.azure.com/api/projects/example",
            "--agent",
            "langchain-azure-resilient-responses-steerable",
            "--tokendelay",
            "0.5",
        ]
    )

    assert args.agent == "langchain-azure-resilient-responses-steerable"
    assert args.tokendelay == "0.5"
