"""Graph schema — node and edge type names, feature ordering helpers.

Feature lists are read from the config (`graph.features`) so the schema is
config-driven. These helpers just expose the names and the index lookup that
the synthetic builder, the mask, and the effects use.
"""

from __future__ import annotations

NODE_TYPES = ("residential", "road", "shelter", "depot", "hazard_zone")

EDGE_TYPES = (
    ("road", "connects", "road"),
    ("residential", "uses", "road"),
    ("road", "leads_to", "shelter"),
    ("hazard_zone", "exposes", "residential"),
    ("hazard_zone", "exposes", "road"),
    ("depot", "supplies", "residential"),
)

EDGE_FEATURES: tuple[str, ...] = (
    "distance",
    "passability",
    "flood_threshold",
    "travel_time",
    "blocked",
)


def node_feature_names(config: dict, node_type: str) -> list[str]:
    return list(config["graph"]["features"][node_type])


def edge_feature_names() -> list[str]:
    return list(EDGE_FEATURES)


def feature_index(names: list[str], feature: str) -> int:
    try:
        return names.index(feature)
    except ValueError as e:
        raise KeyError(f"Feature {feature!r} not in {names}") from e
