"""Lightweight YAML config loader with `extends:` support.

A scenario YAML may set `extends: <path>`; the loader recursively merges the
parent into the child. Child keys override parent keys; nested dicts merge
deeply. Lists are replaced wholesale (no concat).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving `extends:` against the same directory."""
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    parent_ref = cfg.pop("extends", None)
    if parent_ref:
        parent_path = (path.parent / parent_ref).resolve()
        parent_cfg = load_config(parent_path)
        cfg = _deep_merge(parent_cfg, cfg)

    return cfg
