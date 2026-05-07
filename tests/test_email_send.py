"""Offline tests for the Resend-based EmailSender + builder + parsing.

Background: SMTP is blocked outbound on DigitalOcean droplets, so the
production transport is Resend's HTTPS API. Tests use httpx.MockTransport
to verify the request shape without making real network calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from coastal_agent.email_send import (
    EmailSender,
    EmailSendError,
    build_sender_from_settings,
    parse_recipients,
)


# ----------------------------------------------------------------
# parse_recipients
# ----------------------------------------------------------------


def test_parse_recipients_handles_comma_semicolon_and_whitespace() -> None:
    assert parse_recipients("a@x.com, b@y.com; c@z.com") == [
        "a@x.com", "b@y.com", "c@z.com",
    ]


def test_parse_recipients_empty_string_returns_empty_list() -> None:
    assert parse_recipients("") == []
    assert parse_recipients("   ") == []


# ----------------------------------------------------------------
# build_sender_from_settings
# ----------------------------------------------------------------


def test_builder_returns_none_when_mock_mode() -> None:
    s = SimpleNamespace(
        email_mode="mock", resend_api_key="re_x", email_from="f@x.com",
    )
    assert build_sender_from_settings(s) is None


def test_builder_returns_none_when_real_but_missing_credentials() -> None:
    s = SimpleNamespace(
        email_mode="real", resend_api_key="", email_from="",
    )
    assert build_sender_from_settings(s) is None


def test_builder_returns_none_when_only_api_key_present() -> None:
    s = SimpleNamespace(
        email_mode="real", resend_api_key="re_x", email_from="",
    )
    assert build_sender_from_settings(s) is None


def test_builder_constructs_sender_when_real_and_complete() -> None:
    s = SimpleNamespace(
        email_mode="real",
        resend_api_key="re_test_abc123",
        email_from="onboarding@resend.dev",
    )
    sender = build_sender_from_settings(s)
    assert sender is not None
    assert sender.api_key == "re_test_abc123"
    assert sender.from_addr == "onboarding@resend.dev"


# ----------------------------------------------------------------
# EmailSender.send
# ----------------------------------------------------------------


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


def test_send_posts_to_resend_with_correct_payload() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization", "")
        captured["body"] = request.content
        return httpx.Response(200, json={"id": "msg_123abc"})

    sender = EmailSender(
        api_key="re_test_key",
        from_addr="onboarding@resend.dev",
        client=_client(handler),
    )
    sender.send(
        subject="hello", body="world", to=["a@x.com", "b@y.com"],
    )

    assert captured["url"] == "https://api.resend.com/emails"
    assert captured["auth"] == "Bearer re_test_key"
    import json as _json
    payload = _json.loads(captured["body"])
    assert payload["from"] == "onboarding@resend.dev"
    assert payload["to"] == ["a@x.com", "b@y.com"]
    assert payload["subject"] == "hello"
    assert payload["text"] == "world"


def test_send_raises_on_4xx_with_resend_error_message() -> None:
    def handler(request):
        return httpx.Response(
            422,
            json={"name": "validation_error",
                  "message": "to[0] is not a valid email"},
        )

    sender = EmailSender(
        api_key="re_test", from_addr="onboarding@resend.dev",
        client=_client(handler),
    )
    with pytest.raises(EmailSendError, match="HTTP 422.*not a valid email"):
        sender.send(subject="s", body="b", to=["bad-address"])


def test_send_raises_on_network_error() -> None:
    def handler(request):
        raise httpx.ConnectError("simulated network failure")

    sender = EmailSender(
        api_key="re_test", from_addr="onboarding@resend.dev",
        client=_client(handler),
    )
    with pytest.raises(EmailSendError, match="ConnectError"):
        sender.send(subject="s", body="b", to=["a@x.com"])


def test_send_raises_on_no_recipients() -> None:
    sender = EmailSender(
        api_key="re_test", from_addr="onboarding@resend.dev",
    )
    with pytest.raises(EmailSendError, match="no recipients"):
        sender.send(subject="s", body="b", to=[])


def test_send_raises_when_misconfigured() -> None:
    sender = EmailSender(api_key="", from_addr="")
    with pytest.raises(EmailSendError, match="misconfigured"):
        sender.send(subject="s", body="b", to=["a@x.com"])
