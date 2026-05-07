# Coastal Emergency Agent — EGU 2026

Orchestration agent that turns a trained graph-based deep RL policy into a coastal-emergency decision-support system: real-time forecast ingestion, plan-grounded policy inference, LLM-composed operator briefs with citations, and an incident-aware operator dashboard.

Companion artefact for **EGU General Assembly 2026, Vienna** — *Deep Reinforcement Learning for Operational Coastal Emergency Response With AI Agent Orchestration and Human Oversight* ([abstract EGU26-17369](https://meetingorganizer.copernicus.org/EGU26/EGU26-17369.html), [doi:10.5194/egusphere-egu26-17369](https://doi.org/10.5194/egusphere-egu26-17369)). Case study: Lido di Venezia, Italy.

> **Research artefact.** Not certified or deployed for real emergency-response use. No safety claims. Outputs are illustrative; operational decisions remain with human authorities under the Comune di Venezia Piano Comunale di Emergenza (PCE).

---

## Plain-English summary

When a storm threatens a coastal city, the duty officer at the emergency operations centre asks the same two questions every hour: *is this serious enough to act, and if so, what does the plan tell me to do?* Today those questions are answered by a mix of forecast portals, paper plans, phone calls, and experience — only as reliable as the duty officer's knowledge and experience.

This system implements that loop end-to-end as a research prototype. Every hour it pulls the latest sea-level and weather forecasts for Lido di Venezia, evaluates them against a trained decision model that knows the local geography, vulnerable assets, and the seven response actions allowed under the Comune di Venezia *Piano Comunale di Emergenza* (PCE), and decides whether the situation warrants opening an incident. When it does, a language model composes a short structured brief — recommendation, the precise PCE provision that authorises it (e.g. *§4.2.3.1 p57*), the live ISPRA tide-gauge reading, and the operational concerns that should shape the decision. The duty officer gets the brief by email with a click-through to a live dashboard; they remain the decision-maker. 

All data used to train the model are publicly available and accessible online.

---

## What it does

```
   ┌────────────┐     ┌─────────────────┐     ┌──────────────────┐
   │ Open-Meteo │ ──▶ │ GAT-PPO policy  │ ──▶ │   orchestrator   │
   │ Marine +   │     │ (vendored,      │     │ (incident-shaped │
   │ Forecast   │     │  Lido HeteroData│     │  state machine)  │
   └────────────┘     │  + plan mask)   │     └────────┬─────────┘
                      └─────────────────┘              │
                                                       ▼
   ┌────────────┐     ┌─────────────────┐     ┌──────────────────┐
   │  Operator  │ ◀── │  LLM brief      │ ◀── │ 6-tool retrieval │
   │  email     │     │  (OpenRouter)   │     │ (PCE plan, FTS5  │
   └────────────┘     │  JSON + cites   │     │  corpus, ISPRA   │
                      └────────┬────────┘     │  tide gauge, …)  │
                               ▼              └──────────────────┘
                      ┌─────────────────┐
                      │ FastAPI dash    │
                      │ + Caddy/HTTPS   │
                      └─────────────────┘
```

- **Polls** Open-Meteo Marine + standard forecast hourly.
- **Activates** an incident when forecast surge exceeds the trigger threshold (110 cm at Lido).
- **Runs** the vendored `GAT-PPO` policy (session-003 seed-2; see `external/gnn_drl_ews_v003_seed2/VERSION.md`) on the cached Lido graph + global state + plan-derived action mask. Output: 7-action distribution + value.
- **Composes** a structured operator brief via OpenRouter (Claude Sonnet by default; model swap = one env var). Six tools available to the LLM: `get_plan_provision`, `verify_preconditions`, `fetch_forecast_detail`, `query_corpus`, `recall_similar_incidents`, `fetch_live_sea_level` (ISPRA).
- **Cites** every recommendation against PCE plan sections and live data sources. Pydantic-validated schema; markdown-fence stripping for LLM JSON.
- **Notifies** operators at three moments per incident — activation, new distinct alert, stand-down — with deeplinks back to the public dashboard. Two transport modes: real (cloud email API) or mock (renders to dashboard panel).
- **Stands down** when surge is below 100 cm for two consecutive ticks.

---

## Status

| Piece | State |
|---|---|
| Local pytest | 118/118 green on Windows + droplet |
| `coastal-agent.service` | active, ticking hourly on a DigitalOcean droplet (Sydney) |
| `coastal-dashboard.service` | active, served by Caddy + Let's Encrypt over HTTPS |
| Forecast feed | Open-Meteo Marine + standard (live) |
| Sea-level fallback | ISPRA → Open-Meteo nowcast → deterministic stub |
| LLM | OpenRouter (default `anthropic/claude-sonnet-4.6`); kill switches via `LLM_ENABLED=false` and `LLM_MAX_BRIEFS_PER_DAY` cap |
| Email | Real (cloud email API) or mock-to-dashboard, configurable per environment |

Real spend is gated: LLM calls only fire when an incident is open AND the brief composer runs. At Lido that means surge ≥ 110 cm — typically a few times per year during *acqua alta*.

---

## Repo layout

```
coastal_agent/                     # the daemon
├── config.py     model.py    db.py
├── policy.py                     # LivePolicy: vendored GAT-PPO inference
├── weather.py    sea_level.py    # external forecast + tide-gauge fetchers
├── trigger.py    orchestrator.py # incident lifecycle state machine
├── tools.py      brief.py        # 6-tool catalog + Pydantic brief schema
├── llm.py                        # OpenRouter composer with tool-use loop
├── email_send.py                 # operator-email transport
├── scheduler.py  main.py         # apscheduler daemon entrypoint

dashboard/                         # FastAPI read-only HTML + JSON
├── api.py                        # routes (/, /incidents/{id}, /api/…)
└── templates/                    # Jinja2, mobile-first dark theme

external/gnn_drl_ews_v003_seed2/   # vendored DRL snapshot — see VERSION.md
├── runs/seed-2/checkpoint.pt     # trained policy state dict (~400 KB)
├── plans/lido.yaml               # PCE-grounded 7-action set
└── data/cache/lido/              # cached OSM + DEM + ISTAT + ISPRA hazards

deploy/                            # systemd units + .env.example
tests/                             # 118 tests, fully offline (no network)
```

The canonical state of the system is what's in `coastal_agent/` plus this README — the code is the spec.

---

## Quick start (local development)

```bash
# 1. Clone + install (uv handles Python 3.12 + venv)
git clone https://github.com/marcellosano/coastal-emergency-agent-egu2026.git
cd coastal-emergency-agent-egu2026
uv sync --extra dev

# 2. Smoke tests (no network required)
uv run pytest

# 3. Run the dashboard against an empty local DB
STATE_DB_PATH=./local_state.db uv run uvicorn dashboard.api:app --reload
# → http://127.0.0.1:8000

# 4. Run the daemon (will tick once and exit unless DB has scenarios)
STATE_DB_PATH=./local_state.db uv run python -m coastal_agent.main
```

Default `STATE_DB_PATH=/var/lib/coastal-agent/state.db` is the droplet path; override on Windows/macOS as shown.

For live-mode inference (real GAT against the cached Lido graph), install the optional extras:

```bash
uv sync --extra dev --extra live   # adds torch-geometric, geopandas, rasterio, …
```

These extras pull ~1 GB of geospatial native libraries and are only needed if you're regenerating scenarios from the vendored snapshot, not for the runtime daemon.

---

## Deployment

The production deployment is a single 2 GB DigitalOcean droplet (Ubuntu 24.04). Two systemd units: `coastal-agent.service` (writer, single tick on a worker thread) and `coastal-dashboard.service` (reader, FastAPI/uvicorn). Caddy in front for TLS via Let's Encrypt. SQLite at `/var/lib/coastal-agent/state.db` (WAL mode; single writer, multiple readers). Secrets (`/opt/coastal-agent.env`, mode 0640 root:coastal) are off-limits to the repo.

Provisioning steps and unit files are in `deploy/`. The droplet itself is provisioned manually — there's no Terraform yet.

---

## Vendored DRL snapshot

`external/gnn_drl_ews_v003_seed2/` is a **frozen** vendored copy of the trained GAT-PPO policy. Re-vendoring requires a new directory (`_v004/`, etc.) so old snapshots remain reproducible. Full provenance — training session, seed selection rationale, holdout metrics — in `external/gnn_drl_ews_v003_seed2/VERSION.md`.

Vendored data sources:

- **OpenStreetMap** (buildings, roads, amenities) — © OpenStreetMap contributors, ODbL.
- **Tinitaly DEM 10 m** — INGV, CC-BY 4.0.
- **ISTAT census tracts** — © ISTAT.
- **ISPRA tide-gauge records** — © ISPRA. Cached subset for Lido di Venezia.
- **PCE Comune di Venezia** plan references — public-record citations only; no plan PDF is redistributed.

---

## Citation

If you use this work, please cite the EGU 2026 contribution:

> Sano, M., Ferrario, D., Casagrande, S., Vascon, S., Torresan, S., and Critto, A.: *Deep Reinforcement Learning for Operational Coastal Emergency Response With AI Agent Orchestration and Human Oversight*, EGU General Assembly 2026, Vienna, Austria, 3–8 May 2026, EGU26-17369, [doi:10.5194/egusphere-egu26-17369](https://doi.org/10.5194/egusphere-egu26-17369), 2026.

**Affiliations.** CMCC Foundation — Euro-Mediterranean Center on Climate Change, Venice, Italy · Department of Environmental Sciences, Informatics and Statistics (DAIS), Ca' Foscari University of Venice, Italy · European Center for Living Technology (ECLT), Venice, Italy · Griffith University, Australia.

---

## License

License is to be added. Until one is committed, default copyright applies — code is viewable but you do not have rights to copy, modify, or redistribute. Open an issue if you'd like access for a specific use.

---

## Repository maintainer

Dr Marcello Sano — `marcellosano@gmail.com`

The codebase is maintained by Marcello Sano; the EGU 2026 contribution it supports is collaborative — see the citation block above for the full author list.
