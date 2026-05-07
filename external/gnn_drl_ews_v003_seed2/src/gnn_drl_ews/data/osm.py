"""Real-data graph builder for Lido.

Takes the cache populated by `scripts/pull_lido_gis.py` and produces a
HeteroData with the same schema as `data/synthetic.py`:

  residential  : zone nodes (k-means cluster of building footprints)
  road         : key junctions selected by betweenness centrality
  shelter      : OSM amenity=school|hospital|townhall (top-K by capacity proxy)
  depot        : OSM amenity=fire_station + civic anchors as fallback
  hazard_zone  : Adriatic-side coastal buffer + lagoon-side low-elevation slice

All coordinates are projected to EPSG:32632 (UTM 32N) for distance work,
which matches the Tinitaly DEM CRS so we can sample elevation directly
without re-projection per node.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from shapely.geometry import LineString, Point, box
from sklearn.cluster import KMeans
from torch_geometric.data import HeteroData

from ..graph.schema import EDGE_FEATURES, EDGE_TYPES, NODE_TYPES

warnings.filterwarnings("ignore")

UTM = "EPSG:32632"


def build_osm_graph(config: dict) -> HeteroData:
    cache = Path(config["graph"]["cache_dir"])
    if not cache.exists():
        raise FileNotFoundError(
            f"Cache not found at {cache}. Run `python scripts/pull_lido_gis.py` first."
        )

    boundary = gpd.read_file(cache / "boundary.gpkg").to_crs(UTM)
    buildings = gpd.read_file(cache / "buildings.gpkg").to_crs(UTM)
    amenities = gpd.read_file(cache / "amenities.gpkg").to_crs(UTM)
    road_nodes = gpd.read_file(cache / "roads_nodes.gpkg").to_crs(UTM)
    road_edges = gpd.read_file(cache / "roads_edges.gpkg").to_crs(UTM)

    counts = config["graph"]["node_counts"]
    n_zones = int(counts["residential"])
    n_roads = int(counts["road"])
    n_shelters = int(counts["shelter"])
    n_depots = int(counts["depot"])
    n_hazards = int(counts["hazard_zone"])

    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)

    # --- 1. Zone nodes (residential) -----------------------------------
    centroids = np.column_stack(
        [buildings.geometry.centroid.x.values, buildings.geometry.centroid.y.values]
    )
    footprint_area = buildings.geometry.area.values  # m^2

    km = KMeans(n_clusters=n_zones, random_state=seed, n_init=10)
    zone_label = km.fit_predict(centroids)
    zone_xy = np.zeros((n_zones, 2), dtype=np.float64)
    zone_built_m2 = np.zeros(n_zones, dtype=np.float64)
    for k in range(n_zones):
        m = zone_label == k
        # weight by footprint area so the zone "centre" is a centre-of-mass
        w = footprint_area[m]
        if w.sum() > 0:
            zone_xy[k] = (centroids[m] * w[:, None]).sum(axis=0) / w.sum()
        else:
            zone_xy[k] = centroids[m].mean(axis=0)
        zone_built_m2[k] = w.sum()

    # --- 2. Key road junctions -----------------------------------------
    # Use betweenness centrality over the OSM graph to pick the top-K.
    G = _road_networkx(road_nodes, road_edges)
    import networkx as nx
    bc = nx.betweenness_centrality(G, weight="length")
    sorted_nodes = sorted(bc.items(), key=lambda kv: kv[1], reverse=True)
    keep = [n for n, _ in sorted_nodes[: n_roads * 4]]  # top pool
    keep = _spatial_thin(keep, road_nodes, min_sep_m=300.0)[:n_roads]
    road_xy = np.array(
        [
            [
                road_nodes.set_index("osmid").loc[n].geometry.x,
                road_nodes.set_index("osmid").loc[n].geometry.y,
            ]
            for n in keep
        ]
    )
    road_osmids = list(keep)

    # --- 3. Shelters & depots from OSM amenities -----------------------
    if "amenity" not in amenities.columns:
        raise ValueError("amenities.gpkg has no 'amenity' column")
    shelter_pool = amenities[amenities["amenity"].isin(["school", "hospital", "townhall"])]
    depot_pool = amenities[amenities["amenity"].isin(["fire_station"])]

    shelter_xy, shelter_amenity = _pick_amenities(shelter_pool, n_shelters, rng)
    depot_xy, depot_amenity = _pick_amenities(depot_pool, n_depots, rng, fallback_pool=amenities)

    # --- 4. Hazard zones (Adriatic + lagoon) ---------------------------
    hazard_xy = _hazard_zone_centroids(boundary.geometry.iloc[0], n_hazards)

    # --- 5. DEM elevation sampling -------------------------------------
    dem_path = cache / "dem_lido.tif"
    elev = {
        "residential": _sample_dem(dem_path, zone_xy),
        "road":        _sample_dem(dem_path, road_xy),
        "shelter":     _sample_dem(dem_path, shelter_xy),
        "depot":       _sample_dem(dem_path, depot_xy),
        "hazard_zone": _sample_dem(dem_path, hazard_xy),
    }

    # --- 6. Build HeteroData -------------------------------------------
    data = HeteroData()
    pos_dict = {
        "residential": zone_xy,
        "road":        road_xy,
        "shelter":     shelter_xy,
        "depot":       depot_xy,
        "hazard_zone": hazard_xy,
    }

    # Population allocation per residential zone.
    # Two paths controlled by config["demography"]["source"]:
    #   - "census" (preferred): area-weighted-by-footprint allocation of
    #     ISTAT 2021 sezione totals to k-means clusters. Real population
    #     and a real 65+ vulnerable count.
    #   - "footprint_proxy" (fallback): distribute a calibrated total
    #     proportional to built footprint area, vulnerable = flat 15%.
    demography_source = (config.get("demography") or {}).get("source", "footprint_proxy")
    zone_vulnerable: np.ndarray | None = None

    if demography_source == "census":
        from .census import load_lido_census, aggregate_per_cluster
        sezioni = load_lido_census(config)
        zone_pop, zone_vulnerable, diag = aggregate_per_cluster(
            buildings=buildings, zone_label=zone_label, sezioni=sezioni,
        )
        zone_pop = zone_pop.astype(np.float32)
        zone_vulnerable = zone_vulnerable.astype(np.float32)
        print(
            f"[osm.py] census demography: {len(sezioni)} sezioni, "
            f"allocated pop={diag['allocated_pop_total']:.0f} of {diag['ground_truth_pop']} "
            f"({diag['allocated_pop_total']/max(diag['ground_truth_pop'],1):.1%}); "
            f"buildings unjoined={diag['buildings_unjoined']}"
        )
    else:
        fallback_total = float(
            (config.get("demography") or {}).get("fallback_total", 20000.0)
        )
        pop_per_m2 = fallback_total / max(zone_built_m2.sum(), 1.0)
        zone_pop = (zone_built_m2 * pop_per_m2).astype(np.float32)

    for nt in NODE_TYPES:
        n = pos_dict[nt].shape[0]
        feature_names = list(config["graph"]["features"][nt])
        x = _make_node_features(
            nt=nt, n=n, feature_names=feature_names,
            elev=elev[nt],
            zone_pop=zone_pop if nt == "residential" else None,
            zone_vulnerable=zone_vulnerable if nt == "residential" else None,
            config=config, rng=rng,
        )
        data[nt].x = x
        data[nt].feature_names = feature_names
        data[nt].pos = torch.from_numpy(pos_dict[nt].astype(np.float32))
        data[nt].num_nodes = n

    # --- 7. Edges ------------------------------------------------------
    # Each helper returns (edge_index, edge_attr) tensors.
    data[("road", "connects", "road")].edge_index, data[("road", "connects", "road")].edge_attr = \
        _edges_road_road(road_xy, G, road_osmids, road_nodes, config, rng)

    data[("residential", "uses", "road")].edge_index, data[("residential", "uses", "road")].edge_attr = \
        _edges_attach_nearest(zone_xy, road_xy, EDGE_FEATURES, config, rng)

    data[("road", "leads_to", "shelter")].edge_index, data[("road", "leads_to", "shelter")].edge_attr = \
        _edges_attach_nearest_rev(shelter_xy, road_xy, EDGE_FEATURES, config, rng)

    data[("hazard_zone", "exposes", "residential")].edge_index, data[("hazard_zone", "exposes", "residential")].edge_attr = \
        _edges_expose(hazard_xy, zone_xy, radius_m=1500.0, config=config, rng=rng)

    data[("hazard_zone", "exposes", "road")].edge_index, data[("hazard_zone", "exposes", "road")].edge_attr = \
        _edges_expose(hazard_xy, road_xy, radius_m=1500.0, config=config, rng=rng)

    data[("depot", "supplies", "residential")].edge_index, data[("depot", "supplies", "residential")].edge_attr = \
        _edges_attach_nearest_rev(zone_xy, depot_xy, EDGE_FEATURES, config, rng)

    # Make sure every declared edge type exists.
    for etype in EDGE_TYPES:
        if etype not in data.edge_types:
            data[etype].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data[etype].edge_attr = torch.zeros((0, len(EDGE_FEATURES)), dtype=torch.float32)

    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _road_networkx(road_nodes: gpd.GeoDataFrame, road_edges: gpd.GeoDataFrame):
    import networkx as nx
    G = nx.Graph()
    for _, row in road_nodes.iterrows():
        G.add_node(int(row["osmid"]), x=row.geometry.x, y=row.geometry.y)
    length_col = "length" if "length" in road_edges.columns else None
    for _, row in road_edges.iterrows():
        u, v = int(row["u"]), int(row["v"])
        if u in G.nodes and v in G.nodes:
            geom = row.geometry
            length = float(row[length_col]) if length_col else (
                geom.length if isinstance(geom, LineString) else 1.0
            )
            G.add_edge(u, v, length=length)
    return G


def _spatial_thin(node_ids: list, road_nodes: gpd.GeoDataFrame, min_sep_m: float) -> list:
    """Greedy thinning: walk top-ranked nodes, drop any within min_sep_m of one already kept."""
    rn = road_nodes.set_index("osmid")
    kept: list[int] = []
    kept_xy: list[tuple[float, float]] = []
    for n in node_ids:
        if n not in rn.index:
            continue
        x, y = rn.loc[n].geometry.x, rn.loc[n].geometry.y
        if all(np.hypot(x - kx, y - ky) >= min_sep_m for kx, ky in kept_xy):
            kept.append(n)
            kept_xy.append((x, y))
    return kept


def _pick_amenities(
    pool: gpd.GeoDataFrame, n: int, rng: np.random.Generator,
    fallback_pool: gpd.GeoDataFrame | None = None,
):
    if len(pool) >= n:
        sel = pool.iloc[:n]
    elif fallback_pool is not None and len(fallback_pool) > 0:
        # Top up with closest non-overlapping amenities.
        used = set(pool.index)
        extra = fallback_pool[~fallback_pool.index.isin(used)]
        sel = gpd.GeoDataFrame(
            list(pool.itertuples(index=False)) + list(extra.iloc[: n - len(pool)].itertuples(index=False)),
        )
        # Easier: just centroid-pad with random points if still short.
        if len(sel) < n:
            sel = pool.copy()
    else:
        sel = pool

    # Pad with random points inside the boundary if still short.
    while len(sel) < n:
        # We don't have boundary here; just clone existing centroids with jitter.
        if len(sel) == 0:
            raise RuntimeError("No amenities available and no pool to pad from.")
        last = sel.iloc[-1]
        sel = gpd.GeoDataFrame(
            list(sel.itertuples(index=False)) + [last],
        )

    xy = np.column_stack(
        [sel.geometry.centroid.x.values[:n], sel.geometry.centroid.y.values[:n]]
    )
    amenity_kind = (
        list(sel["amenity"].values[:n]) if "amenity" in sel.columns else [None] * n
    )
    return xy, amenity_kind


def _hazard_zone_centroids(boundary, n_hazards: int) -> np.ndarray:
    """Two synthetic hazard zones for Lido:
       - 0: Adriatic side — east of the island spine
       - 1: Lagoon side  — west of the island spine
       For n_hazards > 2, add evenly spaced extras along the spine.
    """
    minx, miny, maxx, maxy = boundary.bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    # Eastward and westward offsets sized to half the island width.
    half_w = (maxx - minx) / 2.0
    pts = [
        (cx + 0.6 * half_w, cy),  # Adriatic / east
        (cx - 0.6 * half_w, cy),  # Lagoon / west
    ]
    if n_hazards > 2:
        for i in range(n_hazards - 2):
            frac = (i + 1) / (n_hazards - 1)
            pts.append((cx, miny + frac * (maxy - miny)))
    return np.array(pts[:n_hazards], dtype=np.float64)


def _sample_dem(dem_path: Path, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample of DEM at given UTM 32N coordinates. Returns elevations
    in metres; NaNs (off-tile or NoData) replaced with 1.0 m as a safe floor."""
    if xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    with rasterio.open(dem_path) as src:
        coords = list(zip(xy[:, 0].tolist(), xy[:, 1].tolist()))
        samples = np.array([v[0] for v in src.sample(coords)], dtype=np.float32)
    samples = np.where(np.isfinite(samples) & (samples > -100), samples, 1.0)
    return samples


def _make_node_features(
    nt: str, n: int, feature_names: list[str],
    elev: np.ndarray,
    zone_pop: np.ndarray | None,
    zone_vulnerable: np.ndarray | None,
    config: dict, rng: np.random.Generator,
) -> torch.Tensor:
    """Build the same per-node feature columns as the synthetic builder, using
    real elevations and zone populations where available."""
    f = np.zeros((n, len(feature_names)), dtype=np.float32)
    road_default = float(config["thresholds"]["road_flood_default"])
    # Convert elevation from metres to centimetres (matches synthetic units
    # so the env's `tide_cm - elev_cm` math stays consistent).
    elev_cm = (elev * 100.0).astype(np.float32)
    for i, name in enumerate(feature_names):
        if name == "elevation":
            f[:, i] = elev_cm
        elif name == "population":
            if zone_pop is not None:
                f[:, i] = zone_pop
            else:
                f[:, i] = 0.0
        elif name == "vulnerable_pop":
            if zone_vulnerable is not None:
                f[:, i] = zone_vulnerable
            elif zone_pop is not None:
                f[:, i] = (zone_pop * 0.15).astype(np.float32)  # flat 15% fallback
            else:
                f[:, i] = 0.0
        elif name in ("water_level", "forecast_water_level"):
            f[:, i] = 0.0
        elif name == "evacuated":
            f[:, i] = 0.0
        elif name == "exposure":
            # Higher exposure for low-elevation residential nodes.
            f[:, i] = np.clip(1.0 - elev_cm / 200.0, 0.0, 1.0).astype(np.float32) if nt == "residential" else 0.0
        elif name == "local_flood_threshold":
            f[:, i] = road_default + rng.uniform(-10, 30, n)
        elif name == "flood_threshold":
            f[:, i] = road_default + rng.uniform(-20, 30, n)
        elif name == "passability":
            f[:, i] = 1.0
        elif name == "blocked":
            f[:, i] = 0.0
        elif name == "capacity":
            f[:, i] = rng.integers(80, 300, n).astype(np.float32)
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
            f[:, i] = np.arange(n) % 2
        else:
            f[:, i] = 0.0
    return torch.from_numpy(f)


# --- edge helpers ----------------------------------------------------------

def _edge_features(rng: np.random.Generator, n_edges: int, config: dict) -> torch.Tensor:
    f = np.zeros((n_edges, len(EDGE_FEATURES)), dtype=np.float32)
    road_default = float(config["thresholds"]["road_flood_default"])
    for i, name in enumerate(EDGE_FEATURES):
        if name == "passability":
            f[:, i] = 1.0
        elif name == "flood_threshold":
            f[:, i] = road_default + rng.uniform(-10, 20, n_edges)
        elif name == "travel_time":
            f[:, i] = rng.uniform(1.0, 5.0, n_edges)
        elif name == "blocked":
            f[:, i] = 0.0
    return torch.from_numpy(f)


def _set_distances(attr: torch.Tensor, src_xy: np.ndarray, dst_xy: np.ndarray,
                   src_idx: np.ndarray, dst_idx: np.ndarray) -> torch.Tensor:
    d = np.hypot(src_xy[src_idx, 0] - dst_xy[dst_idx, 0],
                 src_xy[src_idx, 1] - dst_xy[dst_idx, 1])
    # km
    attr[:, EDGE_FEATURES.index("distance")] = torch.from_numpy((d / 1000.0).astype(np.float32))
    return attr


def _edges_road_road(road_xy, G, road_osmids, road_nodes_gdf, config, rng):
    """Connect kept road nodes by walking shortest paths in the underlying
    OSM graph and collapsing them to single edges in the reduced graph."""
    import networkx as nx
    src_list, dst_list = [], []
    n = len(road_osmids)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                # Distance threshold to avoid pulling all-pairs:
                if np.hypot(road_xy[i, 0] - road_xy[j, 0],
                            road_xy[i, 1] - road_xy[j, 1]) > 3500.0:
                    continue
                _ = nx.shortest_path_length(G, road_osmids[i], road_osmids[j], weight="length")
                src_list.extend([i, j])
                dst_list.extend([j, i])
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, len(EDGE_FEATURES)))
    ei = torch.from_numpy(np.array([src_list, dst_list], dtype=np.int64))
    attr = _edge_features(rng, ei.shape[1], config)
    attr = _set_distances(attr, road_xy, road_xy,
                          np.array(src_list, dtype=np.int64),
                          np.array(dst_list, dtype=np.int64))
    return ei, attr


def _edges_attach_nearest(src_xy: np.ndarray, dst_xy: np.ndarray,
                          _edge_features_unused, config: dict, rng: np.random.Generator):
    """Each src attaches to its nearest dst (one outgoing edge per src)."""
    n = src_xy.shape[0]
    if n == 0 or dst_xy.shape[0] == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, len(EDGE_FEATURES)))
    nearest = _argmin_distance(src_xy, dst_xy)
    src_idx = np.arange(n, dtype=np.int64)
    ei = torch.from_numpy(np.stack([src_idx, nearest]))
    attr = _edge_features(rng, ei.shape[1], config)
    attr = _set_distances(attr, src_xy, dst_xy, src_idx, nearest)
    return ei, attr


def _edges_attach_nearest_rev(targ_xy: np.ndarray, src_xy: np.ndarray,
                              _edge_features_unused, config: dict, rng: np.random.Generator):
    """For each target, draw an edge from its nearest src. Returns
    (src_idx, target_idx) so both axes are correct for the named edge type."""
    n = targ_xy.shape[0]
    if n == 0 or src_xy.shape[0] == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, len(EDGE_FEATURES)))
    nearest_src = _argmin_distance(targ_xy, src_xy)
    targ_idx = np.arange(n, dtype=np.int64)
    ei = torch.from_numpy(np.stack([nearest_src, targ_idx]))
    attr = _edge_features(rng, ei.shape[1], config)
    attr = _set_distances(attr, src_xy, targ_xy, nearest_src, targ_idx)
    return ei, attr


def _edges_expose(hz_xy: np.ndarray, target_xy: np.ndarray, radius_m: float,
                  config: dict, rng: np.random.Generator):
    if hz_xy.shape[0] == 0 or target_xy.shape[0] == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, len(EDGE_FEATURES)))
    src_list, dst_list = [], []
    for h in range(hz_xy.shape[0]):
        d = np.hypot(target_xy[:, 0] - hz_xy[h, 0], target_xy[:, 1] - hz_xy[h, 1])
        for t in np.where(d <= radius_m)[0]:
            src_list.append(h); dst_list.append(int(t))
    if not src_list:
        # Guarantee at least one exposure edge per hazard zone.
        for h in range(hz_xy.shape[0]):
            t = int(np.argmin(np.hypot(target_xy[:, 0] - hz_xy[h, 0],
                                       target_xy[:, 1] - hz_xy[h, 1])))
            src_list.append(h); dst_list.append(t)
    ei = torch.from_numpy(np.array([src_list, dst_list], dtype=np.int64))
    attr = _edge_features(rng, ei.shape[1], config)
    attr = _set_distances(attr, hz_xy, target_xy,
                          np.array(src_list, dtype=np.int64),
                          np.array(dst_list, dtype=np.int64))
    return ei, attr


def _argmin_distance(src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    out = np.empty(src_xy.shape[0], dtype=np.int64)
    for i in range(src_xy.shape[0]):
        d = np.hypot(src_xy[i, 0] - dst_xy[:, 0], src_xy[i, 1] - dst_xy[:, 1])
        out[i] = int(np.argmin(d))
    return out
