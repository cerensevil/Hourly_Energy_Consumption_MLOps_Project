# Energy Forecasting MLOps System

End-to-end hourly energy demand forecasting pipeline with production-grade MLOps infrastructure — experiment tracking, model registry, orchestration, real-time monitoring, and business impact dashboard.

---

## Architecture

```
Raw CSV (10 states)
    │
    ▼
Prefect Orchestration ──► MLflow (experiment tracking + model registry)
    │
    ▼
FastAPI (/predict · /metrics · /available-states)
    │
    ├──► Prometheus (scrape :8000/metrics every 5s)
    │         │
    │         ▼
    │       Grafana (live monitoring dashboard)
    │         │
    │         ▼
    │       Alertmanager (threshold alerts)
    │
    ├──► Inference Flow (2018 replay · 10 states · continuous)
    │
    └──► Streamlit (business dashboard · cost impact)
```

**Stack:** FastAPI · MLflow · Prefect · Prometheus · Grafana · Streamlit · PostgreSQL · Docker Compose

---

## Features

- **Multi-state forecasting** — XGBoost models trained per state (AEP, COMED, DAYTON, DEOK, DOM, DUQ, EKPC, FE, NI, PJME, PJMW)
- **Automated pipeline** — data preparation → training → inference orchestrated via Prefect
- **Experiment tracking** — MLflow model registry with automated versioning and stage promotion
- **Production monitoring** — Prometheus metrics (MAE, RMSE, cost-weighted error, latency, underprediction ratio) per state and model version
- **Business dashboard** — asymmetric cost penalties (underprediction 3×) with up to 86% cost reduction vs. baseline
- **One-command startup** — local and Docker modes via Makefile

---

## Quick Start

### Local mode

```bash
# Start all services
make up-local

# Deploy flows (data prep → training → inference, sequential)
make deploy-local
```

### Docker mode

```bash
# Build and start everything (deployment runs automatically)
make up

# Stop
make down
```

### Services

| Service    | URL                        |
|------------|----------------------------|
| API        | http://localhost:8000       |
| Streamlit  | http://localhost:8501       |
| MLflow     | http://localhost:5000       |
| Prefect    | http://localhost:4200       |
| Prometheus | http://localhost:9091 (local) / :9090 (docker) |
| Grafana    | http://localhost:3000       |

---

## Data Setup

Raw CSV files go in `data/raw_data/`. The data preparation flow processes them into `data/processed/` automatically when you run `make deploy-local`.

Expected format: `{STATE}_hourly.csv` with a `Datetime` column and a MW column.

---

## Project Structure

```
src/
├── api/          # FastAPI app, model loader, feature builder
├── config/       # Central state list (states.py)
├── inference/    # Inference simulation flow
├── metrics/      # Cost-weighted error, evaluation metrics
├── models/       # Model definitions
├── monitoring/   # Prometheus metrics definitions
├── pipeline/     # Data preparation flow
├── registry/     # Model registry artifacts
├── training/     # Training and evaluate-and-promote flow
└── validation/   # Great Expectations data validation

ui/
└── app.py        # Streamlit business dashboard

grafana_data/
└── provisioning/ # Auto-provisioned Prometheus datasource
```

---

## Monitoring Metrics

| Metric | Description |
|---|---|
| `prediction_count_total` | Total predictions per state/model |
| `prediction_latency_seconds` | Inference latency histogram |
| `prediction_absolute_error` | MAE per state |
| `prediction_squared_error` | For RMSE calculation |
| `cost_weighted_error` | Asymmetric business cost metric |
| `underprediction_total` | Underprediction count per state |
| `overprediction_total` | Overprediction count per state |
| `active_model_info` | Currently active model per state |
| `baseline_squared_error` | Lag-24 baseline for comparison |

---

## Prefect Flows

| Deployment | Schedule | Description |
|---|---|---|
| `energy-data-prep` | 02:00 daily | Raw CSV → processed parquet |
| `daily-evaluate-promote` | 00:00 daily | Retrain + promote best model |
| `training-deployment` | manual | On-demand training |
| `inference-simulation` | manual | 2018 continuous replay |

---

## Business Dashboard

Streamlit dashboard simulates 2018 as live production data:

- **Total cost** vs baseline comparison
- **Underprediction rate** — critical risk metric (penalty 3×)
- **Cost over time** — hourly cost distribution
- **Actual vs prediction** — visual model performance
- Adjustable penalty parameters per session

---

## Development

```bash
# API only (hot reload)
make dev

# Health check
make test

# Stop local services
make down-local
```
