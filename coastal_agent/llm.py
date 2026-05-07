"""LLM brief composer (DESIGN §3.5 / agency level 2).

Wraps an OpenAI-compatible chat-completion client (OpenRouter by default,
but anything OpenAI-SDK-compatible works) and runs a tool-use loop:

  1. Send system prompt + user context + tool catalog to the LLM.
  2. If the response contains tool calls, dispatch them via ToolDispatcher,
     send the results back, repeat.
  3. When the LLM produces a final assistant message, parse it as JSON,
     validate against the Brief Pydantic schema. One retry on parse/validate
     failure, then fall back to a stub brief.

Hallucination defenses:
  - temperature 0
  - response_format = json_object (every model that supports it)
  - schema enforced via Pydantic on the parsed JSON
  - mask is authoritative — the LLM is told `monitor` and the recommended
    action are the only legal options to *describe*; it cannot *override*
    the recommendation
  - mandatory tool_calls audit trail in the brief

Kill switches:
  - LLM_ENABLED=false → compose_brief returns a stub immediately
  - LLM_MAX_BRIEFS_PER_DAY → BriefBudget tracks count via DB, returns stub
    when cap is hit

Both kill switches preserve the same brief-shape contract — the stub still
matches the Brief schema, so downstream code (dashboard, email composition)
treats real and stub briefs identically.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import ValidationError

from coastal_agent.brief import (
    Brief,
    Citation,
    PreconditionCheck,
    Recommendation,
    ToolCall,
)
from coastal_agent.policy import confidence_signal as policy_confidence
from coastal_agent.scenario import LIDO_ACTIONS, ScenarioRecord
from coastal_agent.tools import (
    TOOL_NAMES,
    TOOL_SCHEMAS,
    ToolContext,
    ToolDispatcher,
    short_summary,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# OpenAI-SDK-compatible client protocol
# ---------------------------------------------------------------------
# Only the bits we touch — saves us coupling to the openai package
# version and lets tests pass a fake client.


class _ChatCompletions(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _Chat(Protocol):
    completions: _ChatCompletions


class LLMClient(Protocol):
    chat: _Chat


# ---------------------------------------------------------------------
# Daily budget tracker (kill-switch via DB count)
# ---------------------------------------------------------------------


class BriefBudget:
    """Counts non-stub briefs per UTC day. Cheap: queries briefs table."""

    def __init__(self, conn: Any, max_per_day: int) -> None:
        self.conn = conn
        self.max_per_day = max_per_day

    def remaining_today(self) -> int:
        if self.max_per_day <= 0:
            return 0
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM briefs "
            "WHERE date(composed_at) = ? "
            "AND json_extract(brief_json, '$.recommendation.rationale') NOT LIKE '[stub%'",
            (today,),
        ).fetchone()
        used = int(row["c"]) if row is not None else 0
        return max(0, self.max_per_day - used)

    def can_compose(self) -> bool:
        return self.remaining_today() > 0


# ---------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------


SYSTEM_PROMPT = """\
You are the brief composer for a coastal emergency operations agent.

A trained policy has produced a recommended action. Your job: produce a
short structured JSON brief that grounds the recommendation in the case
study's emergency plan, verifies preconditions, and lists concerns the
human operator should consider before acting.

Hard rules:
  1. The recommended action is FIXED — `{recommended_action_id}`. Do not
     suggest a different action. Your job is to *explain and verify*,
     not to second-guess.
  2. Cite real sources. Every non-monitor recommendation must have at
     least one citation pointing at output from `get_plan_provision`,
     `query_corpus`, `recall_similar_incidents`, or `fetch_live_sea_level`.
     If a tool returns no data, do not invent citations.
  3. Use the tool catalog. Call `get_plan_provision` first, then
     `verify_preconditions`. Call other tools only if they would
     genuinely improve the brief.
  4. Output exactly one JSON object matching the schema given in the user
     message. No prose outside the JSON.
"""


USER_PROMPT_TEMPLATE = """\
Case study: {case_study}
Incident: {incident_id} (tick {tick} since opening at {opened_at})
Simulated time: {simulated_time}

Forecast:
  surge_cm: {surge_cm}
  wind_ms: {wind_ms}
  wave_m: {wave_m}
  rainfall_mm: {rainfall_mm}
  horizon_hours: {horizon_hours}

Storm classification: {storm_type}
Confidence signal: {confidence_signal}

Legal action mask (only these may be described as options):
{mask_block}

Action probabilities from policy:
{probs_block}

Recommended action: {recommended_action_id}

Produce a JSON object with these keys:
{{
  "incident_id": <int>,
  "tick": <int>,
  "recommendation": {{
    "action_id": "{recommended_action_id}",
    "confidence_signal": "{confidence_signal}",
    "rationale": "<one paragraph, grounded in tool outputs>"
  }},
  "citations": [
    {{ "source_type": "plan_provision|corpus_chunk|live_sea_level|similar_incident|forecast_detail",
       "ref": "<the ref string returned by the tool>",
       "excerpt": "<optional short excerpt>" }}
  ],
  "precondition_check": [
    {{ "condition": "<from plan>", "satisfied": true|false,
       "evidence": "<why>" }}
  ],
  "concerns": [ "<short concern>" ],
  "open_questions": [ "<short open question>" ],
  "tool_calls": []
}}

Leave `tool_calls` as an empty list — the runtime fills it from the
audit trail of your tool invocations.
"""


@dataclass
class ComposeResult:
    """What `compose_brief` returns to the orchestrator."""

    brief: Brief
    is_stub: bool
    reason: str | None = None
    raw_text: str | None = None
    tool_call_audit: list[ToolCall] = field(default_factory=list)


class LLMComposer:
    """Composes a Brief for one tick.

    Construct once per process. Pass a real OpenAI(-compatible) client
    in production and a fake in tests.
    """

    def __init__(
        self,
        client: LLMClient | None,
        model: str,
        budget: BriefBudget,
        enabled: bool = True,
        max_tool_iterations: int = 8,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.model = model
        self.budget = budget
        self.enabled = enabled
        self.max_tool_iterations = max_tool_iterations
        self.temperature = temperature

    # -- public entry point ------------------------------------------

    def compose_brief(
        self,
        record: ScenarioRecord,
        recommended_action_id: str,
        case_study: str,
        incident_id: int,
        opened_at: datetime,
        ctx: ToolContext,
    ) -> ComposeResult:
        if not self.enabled:
            return self._stub(record, recommended_action_id, incident_id,
                              reason="kill_switch_disabled")
        if self.client is None:
            return self._stub(record, recommended_action_id, incident_id,
                              reason="no_llm_client")
        if not self.budget.can_compose():
            return self._stub(record, recommended_action_id, incident_id,
                              reason="daily_cap_reached")

        dispatcher = ToolDispatcher(ctx)
        try:
            brief, audit, raw = self._run_tool_loop(
                record, recommended_action_id, case_study,
                incident_id, opened_at, dispatcher,
            )
            return ComposeResult(
                brief=brief, is_stub=False, raw_text=raw,
                tool_call_audit=audit,
            )
        except Exception as e:
            log.warning("LLM compose failed: %s; falling back to stub", e)
            return self._stub(
                record, recommended_action_id, incident_id,
                reason=f"compose_error: {type(e).__name__}: {e}",
            )

    # -- internals ----------------------------------------------------

    def _stub(
        self,
        record: ScenarioRecord,
        recommended_action_id: str,
        incident_id: int,
        reason: str,
    ) -> ComposeResult:
        brief = Brief(
            incident_id=incident_id,
            tick=record.tick,
            recommendation=Recommendation(
                action_id=recommended_action_id,
                confidence_signal=policy_confidence(record.policy_output),
                rationale=f"[stub: {reason}]",
            ),
            citations=[],
            precondition_check=[],
            concerns=[],
            open_questions=[],
            tool_calls=[],
        )
        return ComposeResult(brief=brief, is_stub=True, reason=reason)

    def _run_tool_loop(
        self,
        record: ScenarioRecord,
        recommended_action_id: str,
        case_study: str,
        incident_id: int,
        opened_at: datetime,
        dispatcher: ToolDispatcher,
    ) -> tuple[Brief, list[ToolCall], str]:
        sys_prompt = SYSTEM_PROMPT.format(
            recommended_action_id=recommended_action_id,
        )
        user_prompt = self._render_user_prompt(
            record, recommended_action_id, case_study, incident_id, opened_at,
        )

        messages: list[dict] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        audit: list[ToolCall] = []

        for _ in range(self.max_tool_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            choice = response.choices[0]
            message = choice.message

            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                # Append assistant turn (with tool_calls) and the tool results.
                messages.append(_assistant_msg_with_tool_calls(message, tool_calls))
                for tc in tool_calls:
                    name = tc.function.name
                    raw_args = tc.function.arguments or "{}"
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}
                    if name not in TOOL_NAMES:
                        result = {"error": f"unknown tool: {name}"}
                        latency_ms = 0
                    else:
                        result, latency_ms = dispatcher.call(name, args)
                    audit.append(
                        ToolCall(
                            name=name,
                            args=args,
                            result_summary=short_summary(name, result),
                            latency_ms=latency_ms,
                            error=result.get("error") if isinstance(result, dict) else None,
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result),
                        }
                    )
                continue

            # No more tool calls — the assistant produced final JSON.
            raw = message.content or ""
            brief = self._parse_and_validate(
                raw, record, recommended_action_id, incident_id, audit,
            )
            return brief, audit, raw

        raise RuntimeError(
            f"LLM tool loop exceeded {self.max_tool_iterations} iterations"
        )

    def _parse_and_validate(
        self,
        raw: str,
        record: ScenarioRecord,
        recommended_action_id: str,
        incident_id: int,
        audit: list[ToolCall],
    ) -> Brief:
        cleaned = _strip_code_fences(raw)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM output not valid JSON: {e}") from e

        # Force these so the model can't drift even if it tries.
        payload["incident_id"] = incident_id
        payload["tick"] = record.tick
        rec = payload.get("recommendation") or {}
        rec["action_id"] = recommended_action_id
        if "confidence_signal" not in rec or rec["confidence_signal"] not in {
            "high", "medium", "low", "split",
        }:
            rec["confidence_signal"] = policy_confidence(record.policy_output)
        payload["recommendation"] = rec

        # Inject the tool-call audit trail (the model leaves this empty
        # per the prompt; we own this field).
        payload["tool_calls"] = [tc.model_dump() for tc in audit]

        try:
            return Brief.model_validate(payload)
        except ValidationError as e:
            raise ValueError(f"LLM output failed Brief validation: {e}") from e

    def _render_user_prompt(
        self,
        record: ScenarioRecord,
        recommended_action_id: str,
        case_study: str,
        incident_id: int,
        opened_at: datetime,
    ) -> str:
        mask_block = "\n".join(
            f"  {LIDO_ACTIONS[i]}: {'legal' if record.mask[i] else 'illegal'}"
            for i in range(len(LIDO_ACTIONS))
        )
        probs_block = "\n".join(
            f"  {LIDO_ACTIONS[i]}: {record.policy_output.action_probs[i]:.3f}"
            for i in range(len(LIDO_ACTIONS))
        )
        return USER_PROMPT_TEMPLATE.format(
            case_study=case_study,
            incident_id=incident_id,
            tick=record.tick,
            opened_at=opened_at.isoformat(),
            simulated_time=record.simulated_time.isoformat(),
            surge_cm=record.forecast.surge_cm,
            wind_ms=record.forecast.wind_ms,
            wave_m=record.forecast.wave_m,
            rainfall_mm=record.forecast.rainfall_mm,
            horizon_hours=record.forecast.horizon_hours,
            storm_type=record.policy_output.storm_type,
            confidence_signal=policy_confidence(record.policy_output),
            mask_block=mask_block,
            probs_block=probs_block,
            recommended_action_id=recommended_action_id,
        )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n(.*?)\n```",
    re.DOTALL,
)


def _strip_code_fences(s: str) -> str:
    """Extract a JSON payload from an LLM response.

    Claude (and similar) often emits prose before/after a fenced JSON
    block, even when response_format=json_object is requested. Strategy:

      1. If the response contains a ``` fenced block, return the FIRST
         such block's contents (Claude's pattern: "I have all the data.
         \\n```json\\n{...}\\n```").
      2. Otherwise, try the response as-is; the caller's json.loads
         will tell us if it isn't JSON.
    """
    text = s.strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


def _assistant_msg_with_tool_calls(message: Any, tool_calls: list) -> dict:
    """Reconstruct the assistant turn for the next round of messages.

    OpenAI client SDK objects → plain-dict messages compatible with the
    chat.completions API contract.
    """
    return {
        "role": "assistant",
        "content": getattr(message, "content", None),
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ],
    }


def build_default_composer(conn: Any, settings_obj: Any) -> "LLMComposer":
    """Construct a real composer from settings. Returns a no-LLM composer
    if no API key is configured — same Brief contract, stub output.
    """
    budget = BriefBudget(conn, settings_obj.llm_max_briefs_per_day)

    if not settings_obj.llm_enabled or not settings_obj.llm_api_key:
        return LLMComposer(
            client=None,
            model=settings_obj.llm_model,
            budget=budget,
            enabled=False,
        )

    # Lazy import so tests / kill-switched runs don't pay the openai
    # import cost.
    from openai import OpenAI

    client = OpenAI(
        api_key=settings_obj.llm_api_key,
        base_url=settings_obj.llm_base_url,
    )
    return LLMComposer(
        client=client,
        model=settings_obj.llm_model,
        budget=budget,
        enabled=True,
        max_tool_iterations=settings_obj.llm_max_tool_iterations,
        temperature=settings_obj.llm_temperature,
    )
