"""Procedural Lido-shaped graph generator.

Produces a HeteroData graph with the node and edge types declared in
`graph/schema.py`, sized from `config['graph']['node_counts']` and with
features sized from `config['graph']['features'][<node_type>]`.

Geometry is a single north–south spine of road nodes (mimicking Lido's
spine road). Residential, shelter, and depot nodes attach to nearby road
nodes. Hazard zones expose a contiguous slice of residential and road
nodes. This is intentionally simple — the goal at M1 is a runnable
HeteroData object with internally consistent indices.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData

from ..graph.schema import EDGE_FEATURES, EDGE_TYPES, NODE_TYPES


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_node_features(
    rng: np.random.Generator, node_type: str, n: int, feature_names: list[str], config: dict
) -> torch.Tensor:
    """Generate plausible-but-cheap features. All values are floats."""
    f = np.zeros((n, len(feature_names)), dtype=np.float32)
    road_default = float(config["thresholds"].get("road_flood_default", 90.0))

    for i, name in enumerate(feature_names):
        if name == "elevation":
            # cm above local mean. Roads slightly elevated, residential mixed,
            # hazard_zone at zero, shelters higher.
            if node_type == "shelter":
                f[:, i] = rng.uniform(150, 250, n)
            elif node_type == "road":
                f[:, i] = rng.uniform(80, 140, n)
            elif node_type == "residential":
                f[:, i] = rng.uniform(60, 160, n)
            elif node_type == "depot":
                f[:, i] = rng.uniform(120, 200, n)
            else:  # hazard_zone
                f[:, i] = 0.0
        elif name == "population":
            f[:, i] = rng.integers(20, 400, n).astype(np.float32)
        elif name == "vulnerable_pop":
            f[:, i] = (rng.uniform(0.05, 0.30, n) * rng.integers(20, 400, n)).astype(np.float32)
        elif name in ("water_level", "forecast_water_level"):
            f[:, i] = 0.0
        elif name == "evacuated":
            f[:, i] = 0.0
        elif name == "exposure":
            f[:, i] = rng.uniform(0.0, 1.0, n)
        elif name == "local_flood_threshold":
            f[:, i] = road_default + rng.uniform(-10, 30, n)
        elif name == "flood_threshold":
            f[:, i] = road_default + rng.uniform(-20, 30, n)
        elif name == "passability":
            f[:, i] = 1.0
        elif name == "blocked":
            f[:, i] = 0.0
        elif name == "capacity":
            f[:, i] = rng.integers(50, 300, n).astype(np.float32)
        elif name == "occupancy":
            f[:, i] = 0.0
        elif name == "accessible":
            f[:, i] = 1.0
        elif name == "shelter_active":
            f[:, i] = 0.0
        elif name == "resources":
            f[:, i] = float(config["resources"].get("initial_budget", 0))
        elif name == "available":
            f[:, i] = 1.0
        elif name == "intensity":
            f[:, i] = 0.0
        elif name == "type_id":
            # 0 == tide/coastal, 1 == wind/wave, … assigned round-robin.
            f[:, i] = np.arange(n) % 2
        else:
            # Unknown feature name → leave zero. Better than crashing on
            # exploratory configs.
            f[:, i] = 0.0
    return torch.from_numpy(f)


def _attach_nearest(rng: np.random.Generator, src_pos: np.ndarray, dst_pos: np.ndarray) -> np.ndarray:
    """Each src attaches to its nearest dst by 1D position (returns dst index per src)."""
    out = np.empty(src_pos.shape[0], dtype=np.int64)
    for i, p in enumerate(src_pos):
        out[i] = int(np.argmin(np.abs(dst_pos - p)))
    return out


def _edge_features(rng: np.random.Generator, n_edges: int, config: dict, kind: str) -> torch.Tensor:
    """Build [n_edges, len(EDGE_FEATURES)] tensor with sensible defaults."""
    f = np.zeros((n_edges, len(EDGE_FEATURES)), dtype=np.float32)
    road_default = float(config["thresholds"].get("road_flood_default", 90.0))
    for i, name in enumerate(EDGE_FEATURES):
        if name == "distance":
            f[:, i] = rng.uniform(0.1, 1.5, n_edges)
        elif name == "passability":
            f[:, i] = 1.0
        elif name == "flood_threshold":
            f[:, i] = road_default + rng.uniform(-10, 20, n_edges)
        elif name == "travel_time":
            f[:, i] = rng.uniform(1.0, 5.0, n_edges)
        elif name == "blocked":
            f[:, i] = 0.0
    return torch.from_numpy(f)


def build_synthetic_graph(config: dict) -> HeteroData:
    seed = int(config.get("seed", 0))
    rng = _rng(seed)
    counts = config["graph"]["node_counts"]

    data = HeteroData()

    # --- Nodes -----------------------------------------------------------
    positions: dict[str, np.ndarray] = {}
    for nt in NODE_TYPES:
        n = int(counts.get(nt, 0))
        if n == 0:
            continue
        feature_names = list(config["graph"]["features"][nt])
        x = _make_node_features(rng, nt, n, feature_names, config)
        data[nt].x = x
        data[nt].feature_names = feature_names
        # 1D position along the Lido north–south spine, in [0, 1].
        positions[nt] = rng.uniform(0.0, 1.0, n)
        data[nt].pos = torch.from_numpy(positions[nt].astype(np.float32))
        data[nt].num_nodes = n

    # --- Edges -----------------------------------------------------------
    # 1. road <-> road: chain along spine + a couple of redundant shortcuts.
    if "road" in positions:
        order = np.argsort(positions["road"])
        chain_src = order[:-1]
        chain_dst = order[1:]
        # Add ~2 shortcuts for redundancy.
        n_short = min(2, max(0, len(order) - 2))
        if n_short > 0:
            sh_idx = rng.choice(len(order) - 2, size=n_short, replace=False)
            short_src = order[sh_idx]
            short_dst = order[sh_idx + 2]
        else:
            short_src = np.array([], dtype=np.int64)
            short_dst = np.array([], dtype=np.int64)
        src = np.concatenate([chain_src, chain_dst, short_src, short_dst])
        dst = np.concatenate([chain_dst, chain_src, short_dst, short_src])  # bidirectional
        ei = torch.from_numpy(np.stack([src, dst]).astype(np.int64))
        data[("road", "connects", "road")].edge_index = ei
        data[("road", "connects", "road")].edge_attr = _edge_features(rng, ei.shape[1], config, "road_road")

    # 2. residential -> road (each residential to nearest road).
    if "residential" in positions and "road" in positions:
        nearest = _attach_nearest(rng, positions["residential"], positions["road"])
        src = np.arange(len(positions["residential"]), dtype=np.int64)
        ei = torch.from_numpy(np.stack([src, nearest]))
        data[("residential", "uses", "road")].edge_index = ei
        data[("residential", "uses", "road")].edge_attr = _edge_features(rng, ei.shape[1], config, "res_road")

    # 3. road -> shelter (each shelter draws from nearest road).
    if "shelter" in positions and "road" in positions:
        nearest = _attach_nearest(rng, positions["shelter"], positions["road"])
        dst = np.arange(len(positions["shelter"]), dtype=np.int64)
        ei = torch.from_numpy(np.stack([nearest, dst]))
        data[("road", "leads_to", "shelter")].edge_index = ei
        data[("road", "leads_to", "shelter")].edge_attr = _edge_features(rng, ei.shape[1], config, "road_shelter")

    # 4. hazard_zone -> residential / road (contiguous exposure slice).
    if "hazard_zone" in positions and "residential" in positions:
        ei = _expose(rng, positions["hazard_zone"], positions["residential"], width=0.25)
        data[("hazard_zone", "exposes", "residential")].edge_index = ei
        data[("hazard_zone", "exposes", "residential")].edge_attr = _edge_features(rng, ei.shape[1], config, "hz_res")
    if "hazard_zone" in positions and "road" in positions:
        ei = _expose(rng, positions["hazard_zone"], positions["road"], width=0.25)
        data[("hazard_zone", "exposes", "road")].edge_index = ei
        data[("hazard_zone", "exposes", "road")].edge_attr = _edge_features(rng, ei.shape[1], config, "hz_road")

    # 5. depot -> residential (each residential supplied by nearest depot).
    if "depot" in positions and "residential" in positions:
        nearest = _attach_nearest(rng, positions["residential"], positions["depot"])
        dst = np.arange(len(positions["residential"]), dtype=np.int64)
        ei = torch.from_numpy(np.stack([nearest, dst]))
        data[("depot", "supplies", "residential")].edge_index = ei
        data[("depot", "supplies", "residential")].edge_attr = _edge_features(rng, ei.shape[1], config, "depot_res")

    # Make sure every declared edge type exists, even if empty (avoids
    # KeyErrors downstream).
    for etype in EDGE_TYPES:
        if etype not in data.edge_types:
            data[etype].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data[etype].edge_attr = torch.zeros((0, len(EDGE_FEATURES)), dtype=torch.float32)

    return data


def _expose(
    rng: np.random.Generator,
    hz_pos: np.ndarray,
    target_pos: np.ndarray,
    width: float,
) -> torch.Tensor:
    """For each hazard zone, expose targets within `width` along the spine."""
    src_list: list[int] = []
    dst_list: list[int] = []
    for h_idx, hp in enumerate(hz_pos):
        mask = np.abs(target_pos - hp) <= width
        for t_idx in np.where(mask)[0]:
            src_list.append(h_idx)
            dst_list.append(int(t_idx))
    if not src_list:
        # Guarantee at least one exposure edge so the smoke test exercises
        # hazard propagation even when randomness leaves nothing in range.
        src_list = [0]
        dst_list = [int(np.argmin(np.abs(target_pos - hz_pos[0])))]
    return torch.from_numpy(np.stack([np.array(src_list, dtype=np.int64),
                                       np.array(dst_list, dtype=np.int64)]))
