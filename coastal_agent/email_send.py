"""Real email delivery via the Resend HTTPS API.

Background: DigitalOcean blocks outbound SMTP (ports 25/465/587) on
new accounts as anti-spam policy, so smtplib-based delivery does not
work from this droplet. Resend exposes a single HTTPS POST endpoint
on port 443 — same egress path the LLM and weather APIs already use.

The orchestrator writes a `sent_emails` row regardless of mode. In
**mock** mode it records what *would* have been sent and stops; in
**real** mode it also calls `EmailSender.send` and updates
`delivery_status` ('sent' / 'failed:<reason>').

Sender constraints on the Resend free tier:
  - Without a verified sender domain, `from` must be
    `onboarding@resend.dev` and the only deliverable recipient is
    the address tied to the Resend account.
  - With a verified domain (DNS records added in the Resend UI),
    `from` can be any address on that domain and `to` is unrestricted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx


log = logging.getLogger(__name__)

RESEND_ENDPOINT = "https://api.resend.com/emails"
DEFAULT_TIMEOUT_S = 15.0


class EmailSendError(RuntimeError):
    """Raised when a delivery fails. Caller records the reason in
    `sent_emails.delivery_status='failed:<reason>'`."""


@dataclass(frozen=True)
class EmailSender:
    """Stateless Resend HTTPS client."""

    api_key: str
    from_addr: str
    timeout_s: float = DEFAULT_TIMEOUT_S
    endpoint: str = RESEND_ENDPOINT
    client: httpx.Client | None = None  # tests inject a MockTransport client

    def send(self, *, subject: str, body: str, to: list[str]) -> None:
        if not to:
            raise EmailSendError("no recipients")
        if not self.api_key or not self.from_addr:
            raise EmailSendError("EmailSender misconfigured: api_key/from_addr empty")

        payload = {
            "from": self.from_addr,
            "to": list(to),
            "subject": subject,
            "text": body,
        }
        own_client = self.client is None
        client = self.client or httpx.Client(timeout=self.timeout_s)
        try:
            r = client.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Surface the API's own error message (Resend returns
            # {"name": ..., "message": ...} on 4xx).
            detail = _extract_error_detail(e.response)
            raise EmailSendError(
                f"HTTP {e.response.status_code}: {detail}"
            ) from e
        except httpx.HTTPError as e:
            raise EmailSendError(f"{type(e).__name__}: {e}") from e
        finally:
            if own_client:
                client.close()


def _extract_error_detail(response: Any) -> str:
    try:
        body = response.json()
    except (ValueError, json.JSONDecodeError):
        return response.text[:200]
    if isinstance(body, dict):
        return str(body.get("message") or body.get("error") or body)[:200]
    return str(body)[:200]


def build_sender_from_settings(settings_obj: object) -> EmailSender | None:
    """Construct an EmailSender from a Settings instance, or None when
    real mode isn't configured. Caller passes the result to the
    Orchestrator; orchestrator switches mode based on its presence."""
    mode = str(getattr(settings_obj, "email_mode", "mock")).strip().lower()
    if mode != "real":
        return None
    api_key = str(getattr(settings_obj, "resend_api_key", "")).strip()
    from_addr = str(getattr(settings_obj, "email_from", "")).strip()
    if not api_key or not from_addr:
        log.warning(
            "EMAIL_MODE=real but RESEND_API_KEY/EMAIL_FROM missing; "
            "falling back to mock mode"
        )
        return None
    return EmailSender(api_key=api_key, from_addr=from_addr)


def parse_recipients(raw: str) -> list[str]:
    """Comma- or whitespace-separated list of email addresses."""
    if not raw:
        return []
    out: list[str] = []
    for token in raw.replace(";", ",").split(","):
        addr = token.strip()
        if addr:
            out.append(addr)
    return out
