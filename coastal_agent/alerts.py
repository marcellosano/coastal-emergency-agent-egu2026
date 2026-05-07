"""Alert dedup, brief assembly, dispatch.

Crossing the alert threshold triggers:
  1. Dedup check against recent alerts (same dedup_key in last N hours)
  2. Layered context assembly (DESIGN §12: cached corpus + retrieved + live)
  3. LLM brief generation via `llm.py`
  4. Persist alert row with `brief_text` + `citations_json`
  5. (Phase 6) email dispatch

Phase 4 implements steps 1–2; Phase 5 adds 3–4.
"""

from __future__ import annotations


def maybe_fire(evaluation_id: int) -> None:
    raise NotImplementedError("Phase 4 (dedup) + Phase 5 (brief)")
