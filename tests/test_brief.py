"""Tests for the Brief Pydantic schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from coastal_agent.brief import (
    Brief,
    Citation,
    PreconditionCheck,
    Recommendation,
    ToolCall,
)


def _minimal_payload() -> dict:
    return {
        "incident_id": 1,
        "tick": 7,
        "recommendation": {
            "action_id": "issue_alert",
            "confidence_signal": "high",
            "rationale": "surge crossed activation threshold; PCE §4.1.4.9 mandates alert.",
        },
        "citations": [],
        "precondition_check": [],
        "concerns": [],
        "open_questions": [],
        "tool_calls": [],
    }


def test_minimal_brief_validates() -> None:
    brief = Brief.model_validate(_minimal_payload())
    assert brief.incident_id == 1
    assert brief.tick == 7
    assert brief.recommendation.action_id == "issue_alert"
    assert brief.recommendation.confidence_signal == "high"
    assert brief.citations == []


def test_recommendation_rejects_invalid_confidence_signal() -> None:
    payload = _minimal_payload()
    payload["recommendation"]["confidence_signal"] = "definitely"
    with pytest.raises(ValidationError):
        Brief.model_validate(payload)


def test_recommendation_requires_non_empty_rationale() -> None:
    payload = _minimal_payload()
    payload["recommendation"]["rationale"] = ""
    with pytest.raises(ValidationError):
        Brief.model_validate(payload)


def test_citation_validates() -> None:
    c = Citation(
        source_type="plan_provision",
        ref="PCE_Comune_Venezia_v1_2008 §4.1.4.9 p41",
        excerpt="Sindaco issues public alert via radio/TV/SMS.",
    )
    assert c.source_type == "plan_provision"


def test_citation_rejects_unknown_source_type() -> None:
    with pytest.raises(ValidationError):
        Citation(source_type="hearsay", ref="vibes")


def test_precondition_check_basic() -> None:
    pc = PreconditionCheck(
        condition="global.forecast_tide >= threshold.alert_tide",
        satisfied=True,
        evidence="forecast.surge_cm=112 vs alert_tide=110",
    )
    assert pc.satisfied is True


def test_tool_call_audit_optional_fields() -> None:
    tc = ToolCall(
        name="get_plan_provision",
        args={"action_id": "issue_alert", "case_study": "lido"},
        result_summary="plan_provision issue_alert: PCE §4.1.4.9 p41",
    )
    assert tc.latency_ms is None
    assert tc.error is None


def test_full_brief_with_citations_and_concerns() -> None:
    payload = _minimal_payload()
    payload["citations"] = [
        {
            "source_type": "plan_provision",
            "ref": "PCE §4.1.4.9 p41",
        }
    ]
    payload["concerns"] = ["Vaporetto schedule may be disrupted at high tide."]
    payload["open_questions"] = ["Is the Lido COC currently staffed?"]
    brief = Brief.model_validate(payload)
    assert len(brief.citations) == 1
    assert len(brief.concerns) == 1
    assert len(brief.open_questions) == 1
