"""Lido census demographics (ISTAT 2021).

Two responsibilities:

1. `load_lido_census(config)` — read the Lido-clipped sezioni GeoDataFrame
   produced by `scripts/pull_lido_census.py`. Cached at module level so
   repeated env resets don't pay disk I/O.

2. `aggregate_per_cluster(buildings, zone_label, sezioni)` — given the
   k-means cluster assignment already used by `data/osm.py`, allocate
   ISTAT population to each cluster by tying every building to its
   sezione, then weighting by footprint area within the sezione:

       cluster_pop_k = sum over buildings b in cluster k of
                         weight_b / total_weight_in_sezione_b * P_sezione_b

   Same formula for `vulnerable_pop` using P27+P28+P29.

   This is more honest than naive area-weighted polygon intersection:
   it acknowledges population lives in buildings, not on beaches. A
   sezione containing 50% beach and 50% buildings still allocates 100%
   of its population to the buildings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

_CACHED: dict[str, gpd.GeoDataFrame] = {}


def load_lido_census(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Return Lido sezioni with [SEZ21_ID, geometry, pop_total, pop_65plus,
    POP21, FAM21, EDI21, ...]. Lazy-cached per cache_dir."""
    cache_dir = Path(config["demography"].get("cache_dir") or "data/cache/lido/census")
    gpkg = cache_dir / "sezioni_lido.gpkg"
    key = str(gpkg.resolve())
    if key in _CACHED:
        return _CACHED[key]
    if not gpkg.exists():
        raise FileNotFoundError(
            f"Census cache not found at {gpkg}. "
            f"Run `python scripts/pull_lido_census.py` first."
        )
    gdf = gpd.read_file(gpkg)
    _CACHED[key] = gdf
    return gdf


def aggregate_per_cluster(
    buildings: gpd.GeoDataFrame,
    zone_label: np.ndarray,
    sezioni: gpd.GeoDataFrame,
    *,
    pop_col: str = "pop_total",
    vuln_col: str = "pop_65plus",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Per-building → per-cluster population allocation.

    Args:
        buildings: GeoDataFrame of OSM building footprints (in same CRS as
                   sezioni — UTM 32N). One row per building.
        zone_label: int array, one entry per building, giving the k-means
                    cluster id (0..n_zones-1).
        sezioni:   GeoDataFrame from `load_lido_census`.
        pop_col, vuln_col: which columns of sezioni to allocate.

    Returns:
        (zone_pop, zone_vulnerable, diagnostics)
        zone_pop:        shape [n_zones], float32 — total people per cluster
        zone_vulnerable: shape [n_zones], float32 — people aged 65+ per cluster
        diagnostics:     dict with totals, footprint coverage, etc.
    """
    if len(buildings) != len(zone_label):
        raise ValueError(
            f"buildings ({len(buildings)}) and zone_label ({len(zone_label)}) "
            f"must have the same length"
        )
    if buildings.crs is None or sezioni.crs is None:
        raise ValueError("Both buildings and sezioni must have a CRS set.")
    if buildings.crs != sezioni.crs:
        sezioni = sezioni.to_crs(buildings.crs)

    n_zones = int(zone_label.max()) + 1 if len(zone_label) > 0 else 0

    # 1. Spatially join each building centroid to its sezione.
    centroids = gpd.GeoDataFrame(
        {
            "_idx": np.arange(len(buildings), dtype=np.int64),
            "_label": zone_label.astype(np.int64),
            "_area": buildings.geometry.area.values.astype(np.float64),
        },
        geometry=buildings.geometry.centroid,
        crs=buildings.crs,
    )
    cols_to_keep = ["geometry", "SEZ21_ID", pop_col, vuln_col]
    sj = gpd.sjoin(
        centroids,
        sezioni[cols_to_keep],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"])

    # Drop buildings that didn't fall in any sezione (edge cases at boundary).
    n_unjoined = int(sj["SEZ21_ID"].isna().sum())
    sj = sj.dropna(subset=["SEZ21_ID"])

    # 2. For each sezione, total footprint area within it.
    total_per_sez = sj.groupby("SEZ21_ID")["_area"].sum().rename("_total_area_in_sez")
    sj = sj.merge(total_per_sez, on="SEZ21_ID", how="left")

    # 3. Per-building share of its sezione's population.
    #    If P1 is missing for that sezione, share is 0 (logged via diagnostics).
    sj["_share"] = sj["_area"] / sj["_total_area_in_sez"]
    sj["_pop_share"] = sj["_share"] * sj[pop_col].fillna(0).astype(float)
    sj["_vuln_share"] = sj["_share"] * sj[vuln_col].fillna(0).astype(float)

    # 4. Aggregate to cluster.
    zone_pop = np.zeros(n_zones, dtype=np.float32)
    zone_vuln = np.zeros(n_zones, dtype=np.float32)
    grouped = sj.groupby("_label")[["_pop_share", "_vuln_share"]].sum()
    for k, row in grouped.iterrows():
        zone_pop[int(k)] = float(row["_pop_share"])
        zone_vuln[int(k)] = float(row["_vuln_share"])

    diagnostics = {
        "buildings_total": int(len(buildings)),
        "buildings_unjoined": n_unjoined,
        "buildings_in_sezioni_with_no_pop": int(sj[pop_col].isna().sum()),
        "sezioni_touched": int(sj["SEZ21_ID"].nunique()),
        "allocated_pop_total": float(zone_pop.sum()),
        "allocated_vulnerable_total": float(zone_vuln.sum()),
        "ground_truth_pop": int(sezioni[pop_col].sum()),
        "ground_truth_vulnerable": int(sezioni[vuln_col].sum()),
    }
    return zone_pop, zone_vuln, diagnostics
