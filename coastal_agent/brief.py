"""Brief output schema — the structured JSON the LLM must produce.

Matches the existing stub-brief shape from Orchestrator._write_stub_brief
(which is the operative DESIGN §3.5.B contract). Pydantic-validated so
malformed model output triggers one retry then a stub fallback.

Citation refs are deliberately liberal: tool dispatchers tag their results
with stable refs (plan section, document hash + span, gauge URL) and the
brief asks the LLM to echo those refs verbatim. We don't try to police
exact format — we police *presence*: at least one citation per
non-monitor recommendation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


ConfidenceSignal = Literal["high", "medium", "low", "split"]


class Recommendation(BaseModel):
    action_id: str
    confidence_signal: ConfidenceSignal
    rationale: str = Field(min_length=1)


class Citation(BaseModel):
    """A grounded reference into something a tool returned.

    `source_type` distinguishes plan-doctrine refs (PCE section/page),
    corpus-document refs (doc_hash + span), tide-gauge refs (gauge URL),
    and similar-incident refs (incident_id + tick).
    """

    source_type: Literal[
        "plan_provision",
        "corpus_chunk",
        "live_sea_level",
        "similar_incident",
        "forecast_detail",
    ]
    ref: str = Field(min_length=1)
    excerpt: str | None = None


class PreconditionCheck(BaseModel):
    """One precondition row from the plan, plus whether it's satisfied.

    `satisfied` may be None for conditions that depend on node- or
    graph-level live state we can't fully inspect from forecast alone
    (e.g. shelter accessibility, road passability). The brief should
    state explicitly when verification was deferred rather than
    fabricating a yes/no.
    """

    condition: str
    satisfied: bool | None
    evidence: str


class ToolCall(BaseModel):
    """Audit-trail row for one tool call the LLM made.

    `result_summary` is a short one-liner — full tool output stays out
    of the brief (it's already in the LLM's conversation transcript).
    """

    name: str
    args: dict
    result_summary: str
    latency_ms: int | None = None
    error: str | None = None


class Brief(BaseModel):
    """The structured output the LLM produces per tick.

    Stored as JSON in `briefs.brief_json`. Same keys as the stub brief;
    M5 just fills them with grounded content instead of placeholder text.
    """

    incident_id: int
    tick: int
    recommendation: Recommendation
    citations: list[Citation] = Field(default_factory=list)
    precondition_check: list[PreconditionCheck] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @field_validator("citations")
    @classmethod
    def _at_least_one_citation_for_non_monitor(
        cls, v: list[Citation], info  # type: ignore[no-untyped-def]
    ) -> list[Citation]:
        # Citations strongly recommended but not strictly required —
        # corpus may be empty in early deploys. The orchestrator logs
        # zero-citation briefs to audit_log so we can spot-check.
        return v
