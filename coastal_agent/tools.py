"""Tool catalog the brief-composer LLM can invoke (DESIGN §3.5 / agency level 2).

Six tools, all read-only, all returning JSON-serializable dicts:

  1. get_plan_provision      — PCE doctrine for a plan action (lido.yaml)
  2. verify_preconditions    — checks the action's preconditions against current state
  3. fetch_forecast_detail   — forecast snapshot at a given hour offset
  4. query_corpus            — RAG over corpus/<case_study>/ (graceful empty result)
  5. recall_similar_incidents — past closed incidents from the local DB
  6. fetch_live_sea_level    — ISPRA tide gauge stub (real call lands in M6 live mode)

The dispatcher is stateless across calls but carries a ToolContext with
the current tick's record, DB conn, vendor_dir, corpus_dir. Each call
records latency_ms on the returned dict so the LLM's audit trail
(brief.tool_calls) gets the real numbers.

Two tools are deferred to FUTURE_FEATURES.md: get_action_alternatives,
ask_operator. Not in the MVP catalog.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from coastal_agent.scenario import LIDO_ACTIONS, ScenarioRecord


# ---------------------------------------------------------------------
# OpenAI tool schemas (passed to chat.completions.create as `tools=`)
# ---------------------------------------------------------------------


def _function_tool(name: str, description: str, params: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params,
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _function_tool(
        "get_plan_provision",
        "Retrieve the doctrine entry for a plan action: description, "
        "preconditions, required data, authority, and source citation "
        "(document, sections, pages). Use this to ground the recommendation "
        "in the case study's emergency plan. Always call this for the "
        "recommended action before producing the brief.",
        {
            "type": "object",
            "properties": {
                "action_id": {"type": "string", "enum": list(LIDO_ACTIONS)},
                "case_study": {"type": "string"},
            },
            "required": ["action_id", "case_study"],
            "additionalProperties": False,
        },
    ),
    _function_tool(
        "verify_preconditions",
        "Check whether an action's preconditions are satisfied given the "
        "current incident state. Returns each precondition with a "
        "satisfied flag and the evidence used. Use this to populate "
        "brief.precondition_check.",
        {
            "type": "object",
            "properties": {
                "action_id": {"type": "string", "enum": list(LIDO_ACTIONS)},
                "case_study": {"type": "string"},
            },
            "required": ["action_id", "case_study"],
            "additionalProperties": False,
        },
    ),
    _function_tool(
        "fetch_forecast_detail",
        "Get forecast snapshot at the current tick or at a positive hour "
        "offset. Useful for short-horizon look-ahead (peak surge timing, "
        "wind shift). Tick offset 0 returns the current tick's forecast.",
        {
            "type": "object",
            "properties": {
                "tick_offset": {"type": "integer", "minimum": 0, "maximum": 24},
            },
            "required": ["tick_offset"],
            "additionalProperties": False,
        },
    ),
    _function_tool(
        "query_corpus",
        "Hybrid keyword + semantic search over the case-study corpus "
        "(plans, SOPs, post-incident reports). Returns up to k chunks "
        "with their document hash and span offsets so citations remain "
        "resolvable. Empty corpus returns an empty list — that is OK; "
        "rely on plan_provision in that case.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 3},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    _function_tool(
        "recall_similar_incidents",
        "Retrieve up to k past closed incidents from the local database "
        "matching the query. Returns incident_id, opened_at, "
        "trigger_condition, and a short summary. Empty when no past "
        "incidents exist (cold start).",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 3},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    _function_tool(
        "fetch_live_sea_level",
        "Get the latest sea-level reading from the ISPRA tide gauge "
        "configured for this case study (Lido: lido_diga_sud). Useful "
        "when verifying that the forecast is tracking observation. "
        "Returns gauge_id, value_cm, observed_at, and a stable URL "
        "for citation.",
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
]


TOOL_NAMES: tuple[str, ...] = tuple(t["function"]["name"] for t in TOOL_SCHEMAS)


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------


@dataclass
class ToolContext:
    """Per-tick context passed to every tool handler."""

    record: ScenarioRecord
    incident_id: int
    case_study: str
    conn: sqlite3.Connection
    vendor_dir: Path
    corpus_dir: Path
    upcoming_records: list[ScenarioRecord]
    # When True, tools that would otherwise return a stub may make
    # real network calls. Off in tests; on in the daemon.
    live_data: bool = False


class ToolDispatcher:
    """Maps tool name → handler. Caches the plan YAML on first read."""

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx
        self._plan_cache: dict[str, dict] = {}

    # -- public dispatch entry point ----------------------------------

    def call(self, name: str, args: dict) -> tuple[dict, int]:
        """Run a tool. Returns (result_dict, latency_ms)."""
        handler = self._handlers().get(name)
        if handler is None:
            return ({"error": f"unknown tool: {name}"}, 0)
        t0 = time.perf_counter()
        try:
            result = handler(args)
        except Exception as e:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return ({"error": f"{type(e).__name__}: {e}"}, latency_ms)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return (result, latency_ms)

    def _handlers(self) -> dict[str, Callable[[dict], dict]]:
        return {
            "get_plan_provision": self._get_plan_provision,
            "verify_preconditions": self._verify_preconditions,
            "fetch_forecast_detail": self._fetch_forecast_detail,
            "query_corpus": self._query_corpus,
            "recall_similar_incidents": self._recall_similar_incidents,
            "fetch_live_sea_level": self._fetch_live_sea_level,
        }

    # -- tool implementations -----------------------------------------

    def _load_plan(self, case_study: str) -> dict:
        if case_study in self._plan_cache:
            return self._plan_cache[case_study]
        path = self.ctx.vendor_dir / "plans" / f"{case_study}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"plan YAML for case_study={case_study!r} not found at {path}"
            )
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._plan_cache[case_study] = data
        return data

    def _get_plan_provision(self, args: dict) -> dict:
        action_id = args["action_id"]
        case_study = args["case_study"]
        plan = self._load_plan(case_study)
        for action in plan.get("actions", []):
            if action.get("id") == action_id:
                src = action.get("source", {})
                doc = src.get("document", "")
                sections = src.get("sections", [])
                pages = src.get("pages", [])
                ref = f"{doc} §{','.join(sections)} p{','.join(str(p) for p in pages)}"
                return {
                    "action_id": action_id,
                    "case_study": case_study,
                    "description": action.get("description", ""),
                    "preconditions": action.get("preconditions", []),
                    "required_data": action.get("required_data", []),
                    "authority": src.get("authority", ""),
                    "citation_ref": ref,
                    "source": src,
                }
        return {
            "action_id": action_id,
            "case_study": case_study,
            "error": f"action_id {action_id!r} not in plan {case_study!r}",
        }

    def _verify_preconditions(self, args: dict) -> dict:
        action_id = args["action_id"]
        case_study = args["case_study"]
        plan = self._load_plan(case_study)
        match: dict | None = None
        for action in plan.get("actions", []):
            if action.get("id") == action_id:
                match = action
                break
        if match is None:
            return {
                "action_id": action_id,
                "checks": [],
                "error": f"action_id {action_id!r} not in plan",
            }

        # Operate against the available global state. Node-level
        # preconditions can't be fully verified without the live graph,
        # so we mark them needs_inspection rather than satisfied/unsatisfied.
        forecast = self.ctx.record.forecast
        state = self.ctx.record.state
        thresholds = plan.get("thresholds", {}) or {}
        checks = []
        for cond in match.get("preconditions", []) or []:
            cond_str = str(cond)
            satisfied: bool | None = None
            evidence: str
            if "global.forecast_tide" in cond_str and "alert_tide" in cond_str:
                limit = thresholds.get("alert_tide", 110.0)
                satisfied = forecast.surge_cm >= float(limit)
                evidence = f"forecast.surge_cm={forecast.surge_cm} vs alert_tide={limit}"
            elif "global.forecast_tide" in cond_str and "full_evac" in cond_str:
                limit = thresholds.get("full_evac", 150.0)
                satisfied = forecast.surge_cm >= float(limit)
                evidence = f"forecast.surge_cm={forecast.surge_cm} vs full_evac={limit}"
            elif "global.resources" in cond_str and ">=" in cond_str:
                # crude parse: '>= N' on the right
                try:
                    rhs = float(cond_str.split(">=")[-1].strip())
                except ValueError:
                    rhs = 1.0
                satisfied = state.resources >= rhs
                evidence = f"state.resources={state.resources} vs required={rhs}"
            else:
                evidence = (
                    "node- or graph-level condition — needs live graph "
                    "inspection at runtime"
                )
            checks.append(
                {
                    "condition": cond_str,
                    "satisfied": satisfied,
                    "evidence": evidence,
                }
            )
        return {
            "action_id": action_id,
            "case_study": case_study,
            "checks": checks,
        }

    def _fetch_forecast_detail(self, args: dict) -> dict:
        offset = int(args.get("tick_offset", 0))
        if offset == 0:
            r = self.ctx.record
        else:
            target_idx = offset - 1
            if target_idx < 0 or target_idx >= len(self.ctx.upcoming_records):
                return {
                    "tick_offset": offset,
                    "error": "no forecast available at that horizon",
                }
            r = self.ctx.upcoming_records[target_idx]
        return {
            "tick_offset": offset,
            "tick": r.tick,
            "simulated_time": r.simulated_time.isoformat(),
            "surge_cm": r.forecast.surge_cm,
            "wind_ms": r.forecast.wind_ms,
            "wave_m": r.forecast.wave_m,
            "rainfall_mm": r.forecast.rainfall_mm,
            "horizon_hours": r.forecast.horizon_hours,
        }

    def _query_corpus(self, args: dict) -> dict:
        query = str(args["query"])
        k = int(args.get("k", 3))
        # Hybrid retrieval lands in M5+ (post-corpus-population). For now
        # the corpus dirs are placeholders — return empty results with
        # a note so the LLM doesn't hallucinate citations.
        rows = self.ctx.conn.execute(
            "SELECT COUNT(*) AS c FROM chunks WHERE superseded_at IS NULL"
        ).fetchone()
        n_chunks = int(rows["c"]) if rows is not None else 0
        if n_chunks == 0:
            return {
                "query": query,
                "k": k,
                "results": [],
                "note": "corpus is empty; rely on plan_provision for citations",
            }
        # FTS5-only fallback (vec retrieval requires embeddings, deferred).
        like = f"%{query}%"
        rows = self.ctx.conn.execute(
            "SELECT c.id, c.text, c.span_start, c.span_end, "
            "d.hash AS doc_hash, d.title FROM chunks c "
            "JOIN documents d ON d.id = c.document_id "
            "WHERE c.superseded_at IS NULL AND c.text LIKE ? LIMIT ?",
            (like, k),
        ).fetchall()
        results = [
            {
                "doc_hash": r["doc_hash"],
                "title": r["title"],
                "span": [r["span_start"], r["span_end"]],
                "excerpt": r["text"][:280],
            }
            for r in rows
        ]
        return {"query": query, "k": k, "results": results}

    def _recall_similar_incidents(self, args: dict) -> dict:
        query = str(args.get("query", ""))
        k = int(args.get("k", 3))
        # Most recent closed incidents in the same case_study,
        # excluding the current one.
        rows = self.ctx.conn.execute(
            "SELECT id, opened_at, closed_at, trigger_condition, notes "
            "FROM incidents WHERE case_study = ? AND id != ? "
            "AND status = 'closed' ORDER BY closed_at DESC LIMIT ?",
            (self.ctx.case_study, self.ctx.incident_id, k),
        ).fetchall()
        results = []
        for r in rows:
            try:
                trig = json.loads(r["trigger_condition"]) if r["trigger_condition"] else {}
            except json.JSONDecodeError:
                trig = {}
            results.append(
                {
                    "incident_id": int(r["id"]),
                    "opened_at": r["opened_at"],
                    "closed_at": r["closed_at"],
                    "trigger_summary": trig.get("summary", ""),
                    "ref": f"incident:{r['id']}",
                }
            )
        return {"query": query, "k": k, "results": results}

    def _fetch_live_sea_level(self, args: dict) -> dict:
        forecast = self.ctx.record.forecast
        if not self.ctx.live_data:
            # Test / replay mode — return a deterministic value keyed
            # off the current forecast so the LLM has a coherent number
            # to cite. The stub is tagged so the brief flags it.
            return {
                "gauge_id": "lido_diga_sud",
                "value_cm": forecast.surge_cm,
                "observed_at": self.ctx.record.simulated_time.isoformat(),
                "url": "https://www.mareografico.it/it/i-mareografi/lido-diga-sud",
                "source": "stub",
                "note": "live_data is off (test or replay mode)",
            }
        # Live mode — fetch real observation. Late import so non-live
        # contexts don't pay the import cost.
        from coastal_agent.sea_level import fetch_lido_sea_level

        obs = fetch_lido_sea_level(fallback_value_cm=forecast.surge_cm)
        return {
            "gauge_id": obs.gauge_id,
            "value_cm": obs.value_cm,
            "observed_at": obs.observed_at.isoformat(),
            "url": obs.url or "",
            "source": obs.source,
        }


def short_summary(name: str, result: dict, max_len: int = 200) -> str:
    """One-line audit summary of a tool result for brief.tool_calls."""
    if "error" in result:
        return f"{name} error: {result['error']}"[:max_len]
    if name == "get_plan_provision":
        ref = result.get("citation_ref", "")
        return f"plan_provision {result.get('action_id', '?')}: {ref}"[:max_len]
    if name == "verify_preconditions":
        n = len(result.get("checks", []))
        sat = sum(1 for c in result.get("checks", []) if c.get("satisfied") is True)
        return f"verify_preconditions {result.get('action_id', '?')}: {sat}/{n} satisfied"[:max_len]
    if name == "fetch_forecast_detail":
        return (
            f"forecast_detail t+{result.get('tick_offset', '?')}h: "
            f"surge={result.get('surge_cm', '?')}cm "
            f"wind={result.get('wind_ms', '?')}m/s"
        )[:max_len]
    if name == "query_corpus":
        return f"corpus query '{result.get('query','')}': {len(result.get('results', []))} hits"[:max_len]
    if name == "recall_similar_incidents":
        return f"similar_incidents '{result.get('query','')}': {len(result.get('results', []))} hits"[:max_len]
    if name == "fetch_live_sea_level":
        return f"live_sea_level: {result.get('value_cm','?')}cm at {result.get('gauge_id','?')}"[:max_len]
    return json.dumps(result)[:max_len]
