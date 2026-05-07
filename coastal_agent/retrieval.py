"""Layered retrieval over the case-study corpus (DESIGN §12).

Indexes `corpus/<case_study>/` documents into the `documents` and `chunks`
tables. Hybrid retrieval combines FTS5 (BM25) with sqlite-vec (dense
embeddings) using reciprocal rank fusion.

Idempotent: re-running `index()` only re-embeds documents whose hash has
changed; old chunks are soft-deleted via `superseded_at` so prior
citations remain resolvable.

Phase 5 (post-Phase-3 deploy).
"""

from __future__ import annotations

from pathlib import Path


def index(corpus_dir: Path, case_study: str) -> int:
    """Index a corpus directory. Returns number of chunks (re)embedded."""
    raise NotImplementedError("Phase 5")


def retrieve(query: str, case_study: str, k: int = 5) -> list[dict]:
    """Hybrid retrieval. Returns top-k chunks with doc_hash + spans."""
    raise NotImplementedError("Phase 5")
