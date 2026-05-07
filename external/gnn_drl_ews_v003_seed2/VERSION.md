# Vendored snapshot — `gnn_drl_ews` session-003-reward-shaping iter-001 seed-2

**This is a frozen vendored snapshot. Do not edit.** Updates require a new vendoring with a new directory name (`_v004`, `_v005`, …). Editing these files in place breaks reproducibility — you'd no longer know what the snapshot represents.

## Provenance

| Field | Value |
|---|---|
| **Training session** | `session-003-reward-shaping` |
| **Iteration** | `iter-001` |
| **Seed** | `2` (best-of-3 by operational success; seeds 0 and 1 collapsed to a degenerate mode in this session — do not mix them) |
| **Episode count** | 3000 |
| **Total env-steps** | ~500 000 |
| **Reward shaping** | `compliance` weight = 1.0 |
| **Holdout metrics (300 episodes)** | compliance = 0.91 · success = 0.90 · return = −230 |
| **Date vendored** | 2026-05-06 |
| **Coupling pattern** | Path A — frozen snapshot, not a live dependency |

## Layout

```
gnn_drl_ews_v003_seed2/
├── VERSION.md                     ← this file
├── pyproject.upstream.toml        ← copy of upstream pyproject for reference
├── src/gnn_drl_ews/               ← the Python package (importable)
│   ├── policy/gat_actor_critic.py     model definition
│   ├── env.py                          EWSEnv (graph + state + mask)
│   ├── plan/{loader,mask,effects}.py   plan YAML loader, mask DSL, effects
│   ├── data/{builder,osm,...}.py       graph construction from cached GIS
│   ├── graph/                          schema + transforms
│   ├── hazards/                        synthetic + real-data tide generators
│   ├── config.py                       YAML loader with `extends:` inheritance
│   └── ...
├── runs/seed-2/checkpoint.pt          the trained policy state-dict (~400 KB)
├── runs/seed-2/metrics.json           training metrics summary
├── runs/applied/lido_real.yaml        scenario config used at training time
├── plans/lido.yaml                    PCE-grounded action set (7 actions)
├── plans/schema.yaml                  mask-DSL schema
└── data/cache/lido/                   cached real Lido GIS data
    ├── boundary.gpkg, buildings.gpkg, roads_*.gpkg  (OSM)
    ├── amenities.gpkg                                (OSM POIs)
    ├── dem_lido.tif                                  (Tinitaly 10 m DEM)
    ├── census/                                       (ISTAT)
    └── hazards/                                      (ISPRA tide records)
```

## How this is used

For the EGU demo, the `coastal_agent` runtime loads this snapshot to run live GAT-PPO inference per tick: `EWSEnv.reset_cached()` builds a `HeteroData` graph from the cached GIS files (OSM gpkg + DEM + census + hazard records), and `GATActorCritic.forward(graph, state, mask)` produces the action distribution + value. See `coastal_agent/policy.py` for the adapter.

The runtime code in `coastal_agent/` requires the optional `live` dependency group (`torch-geometric`, `geopandas`, `rasterio`, `gymnasium`, `osmnx`, `scikit-learn`) to load this snapshot. Tests run in replay mode and do not need those deps.

## Caveat on seed selection

Seed 2 is the operationally-good seed from this training session. Seed 0 produced near-uniform output; seed 1 collapsed to ~0.25 compliance. If you ever swap the snapshot for a new training run, **always check the seed's holdout metrics first** — `runs/seed-2/metrics.json` shows what good looks like for this architecture.

## Re-vendoring

If a future training run produces a better policy:

1. Stop using this directory.
2. Vendor the new run as `external/gnn_drl_ews_v004/` (or appropriate version-tagged name).
3. Update `coastal_agent/policy.py` to point at the new path.
4. Add a `VERSION.md` to the new directory mirroring this one.

Do **not** delete this snapshot — keeping old snapshots is cheap (22 MB) and lets reviewers reproduce historical results.
