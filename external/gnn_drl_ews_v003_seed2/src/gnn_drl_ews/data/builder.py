"""Graph builder dispatcher.

The contract from HANDOFF §6:
    build_graph(config: dict) -> HeteroData

Reads `config["graph"]["source"]` and routes to the synthetic or real
builder. M1 only implements the synthetic path; the real branch raises
NotImplementedError so M3 can plug in OSM/DEM/census without changing the
caller.
"""

from __future__ import annotations

from torch_geometric.data import HeteroData


def build_graph(config: dict) -> HeteroData:
    source = config["graph"]["source"]
    if source == "synthetic":
        from .synthetic import build_synthetic_graph
        return build_synthetic_graph(config)
    if source == "osm":
        from .osm import build_osm_graph  # M3
        return build_osm_graph(config)
    raise ValueError(f"Unknown graph.source: {source!r}")
