from client import _invocations_url, build_parser


def test_invocations_arguments_are_accepted() -> None:
    args = build_parser().parse_args(
        [
            "--url",
            "https://example.test/invocations",
            "--auth",
            "--reconnect-timeout",
            "300",
        ]
    )

    assert args.url == "https://example.test/invocations"
    assert args.auth is True
    assert args.reconnect_timeout == 300.0


def test_invocations_argument_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.url == "http://127.0.0.1:8088"
    assert args.auth is False
    assert args.reconnect_timeout == 120.0


def test_invocations_url_accepts_host_or_full_endpoint() -> None:
    assert _invocations_url("http://127.0.0.1:8088") == (
        "http://127.0.0.1:8088/invocations"
    )
    assert _invocations_url("https://example.test/invocations") == (
        "https://example.test/invocations"
    )
    assert _invocations_url(
        "https://example.test/endpoint/protocols/invocations?api-version=v1"
    ) == (
        "https://example.test/endpoint/protocols/invocations?api-version=v1"
    )
