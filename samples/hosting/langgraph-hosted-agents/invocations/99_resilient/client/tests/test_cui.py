from client import build_parser


def test_invocations_arguments_are_accepted() -> None:
    args = build_parser().parse_args(
        [
            "--url",
            "https://example.test/invocations",
            "--session-id",
            "trip-demo",
            "--auth",
            "--reconnect-timeout",
            "300",
        ]
    )

    assert args.url == "https://example.test/invocations"
    assert args.session_id == "trip-demo"
    assert args.auth is True
    assert args.reconnect_timeout == 300.0


def test_invocations_argument_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.url == "http://127.0.0.1:8088/invocations"
    assert args.session_id is None
    assert args.auth is False
    assert args.reconnect_timeout == 120.0
