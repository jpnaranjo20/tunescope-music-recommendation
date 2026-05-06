# tunescope

A music recommendation microservice demonstrating MLOps patterns end-to-end:
MLflow-tracked training, FastAPI serving with hot model reload, blue/green
Docker deploys via a Python load balancer, env-flag A/B testing with per-variant
Prometheus metrics, and a Grafana dashboard — all `docker compose up` on
localhost.

> **Status:** in active development. This README will be filled out (architecture
> diagram, screenshots, full quick-start) once the stack runs end-to-end.

## Original work

This repository is original work authored from scratch. It is **not** a fork or
copy of any school project. It is licensed MIT (see [LICENSE](LICENSE) once
added).

## Planned quick-start

```bash
# Generate synthetic data (no Kaggle download required)
uv run python scripts/generate_synthetic.py

# Train a model into the blue artifact directory
uv run python -m tunescope.models.pipeline --model lgbm_ranker --color blue

# Run the full stack
docker compose up -d --build
```

Open:

- <http://localhost:8080/recommend/42> — recommendations API (via load balancer)
- <http://localhost:8501> — Streamlit UI
- <http://localhost:3000> — Grafana (admin/admin)
- <http://localhost:5000> — MLflow

## Data sources

- **Primary:** Last.fm 1K Users (Òscar Celma research dataset, mirrored on
  Kaggle). Fetched on demand by `scripts/download.sh` into `data/raw/lastfm-1k/`
  (gitignored — never committed).
- **Fallback:** `scripts/generate_synthetic.py` produces a deterministic
  synthetic dataset (~5k users, ~5k tracks, ~500k plays). Used by CI and by
  anyone running the demo without the Kaggle download.
