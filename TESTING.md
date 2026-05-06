# Manual testing log

A running checklist of how to manually verify each piece of the stack as
it lands. Grows as we build. The automated test suite (`uv run pytest`)
covers correctness; this file covers "did it actually feel right when I
ran it."

Each section maps to a checkpoint — once a section's checks pass, that
piece is locked in and we move to the next one.

---

## Checkpoint 1 — Repo bootstrap + synthetic data generator

**Status:** built (commits `98cc7c3`, `6bf4334`).

### What this checkpoint should do

Generate a deterministic synthetic listening dataset (users / tracks /
plays parquet) that downstream code can train on without a Kaggle
download. Locks in determinism, schemas, and per-user timestamp ordering.

### Tests to run

#### 1. Automated suite is green

```bash
uv run pytest -v
```

Expect 6 passing tests covering schemas, row counts, no-orphan IDs,
monotonic timestamps, determinism, and seed-sensitivity.

#### 2. Generator runs at default size

```bash
uv run python scripts/generate_synthetic.py
```

Writes `data/raw/synthetic/{users,tracks,plays}.parquet` (~5k users,
5k tracks, 500k plays). Should take a few seconds. Files are gitignored.

#### 3. Output looks like a music dataset

```bash
uv run python -c "
import pandas as pd
plays = pd.read_parquet('data/raw/synthetic/plays.parquet')
users = pd.read_parquet('data/raw/synthetic/users.parquet')
print('plays head:'); print(plays.head())
print('plays per user (top 10):'); print(plays.user_id.value_counts().head(10))
print('plays per track (top 10):'); print(plays.track_id.value_counts().head(10))
print('time span:', plays.ts.min(), '->', plays.ts.max())
print('countries:'); print(users.country.value_counts())
"
```

Expect:

- power-law shape (a few users / tracks dominate the counts)
- time range spans 2022-01-01 → 2024-01-01 UTC
- 10 country codes roughly balanced

#### 4. Determinism holds across runs

```bash
uv run python scripts/generate_synthetic.py --out-dir /tmp/run-a --seed 99 --plays 10000
uv run python scripts/generate_synthetic.py --out-dir /tmp/run-b --seed 99 --plays 10000
diff <(shasum /tmp/run-a/*.parquet | awk '{print $1}') <(shasum /tmp/run-b/*.parquet | awk '{print $1}')
```

Empty diff = byte-identical output.

#### 5. Lint + format are clean

```bash
uv run ruff check .
uv run ruff format --check .
```

#### 6. CLI flags work

```bash
uv run python scripts/generate_synthetic.py --help
uv run python scripts/generate_synthetic.py --users 50 --tracks 100 --plays 500 --out-dir /tmp/tiny
```

#### 7. Git history is sane

```bash
git log --stat
git ls-files
```

Two commits, no `data/` or `.venv/` tracked.

#### 8. EDA notebook runs end-to-end

```bash
# In VS Code / Jupyter, open:
explore.ipynb
# Then "Run all". Expect plots for plays-per-user, plays-per-track,
# daily volume, two user timelines, and two artist-coherence panels —
# plus the printed numeric summaries beside each.
```

Use the findings to decide whether the generator's distributional
choices need tweaking before we build features off them.

### Not yet testable at this checkpoint

- Last.fm 1K download path (`scripts/download.sh` — not yet built)
- Loader / feature encoding (next checkpoint)
- Anything model-, API-, or Docker-shaped

---

<!-- Future checkpoints get appended below as they land:
## Checkpoint 2 — Data loader + feature encoder
## Checkpoint 3 — LightGBM ranker + MLflow training pipeline
## Checkpoint 4 — FastAPI serving layer
## Checkpoint 5 — Python load balancer + docker-compose
## Checkpoint 6 — Event replayer + online evaluation + Grafana
## Checkpoint 7 — Streamlit UI
-->
