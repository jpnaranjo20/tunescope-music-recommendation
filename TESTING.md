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

## Checkpoint 2 — Data loader + feature encoder

**Status:** built (see commits).

### What this checkpoint should do

Load the parquet output of the synthetic generator (or, when present,
Last.fm 1K) into typed in-memory DataFrames, drop cold-start users
(`<5` plays), produce a leakage-free time-based per-user train/test
split, build the lookup tables a LightGBM ranker needs, and serialize
the encoder to a joblib pickle.

### Tests to run

#### 1. Automated suite is green

```bash
uv run pytest -v
```

Expect 21 passing tests: 7 synthetic + 3 loader + 11 encoder.

#### 2. Loader smoke

```bash
uv run python -c "
from tunescope.data.loader import load
ds = load('data/raw/synthetic')
print('users :', ds.users.shape)
print('tracks:', ds.tracks.shape)
print('plays :', ds.plays.shape)
"
```

Expect three shapes; users count slightly less than 5,000 (cold-start
filter applied), plays count slightly less than 500,000.

#### 3. End-to-end encoder pipeline

```bash
uv run python -c "
import joblib, os
from tunescope.data.loader import load
from tunescope.features.encode import (
    build_labels, fit_encoder, sample_negatives, split_by_time, transform,
)
ds = load('data/raw/synthetic')
train, test = split_by_time(ds.plays, holdout_frac=0.2)
encoder = fit_encoder(train, ds.users, ds.tracks)
positives = build_labels(train, threshold=3)
negatives = sample_negatives(encoder, positives, k=4, seed=0)
pairs = positives[['user_id', 'track_id']].head(50)
features = transform(encoder, pairs)
print('train plays :', len(train))
print('test plays  :', len(test))
print('positives   :', len(positives))
print('negatives   :', len(negatives))
print('feature cols:', list(features.columns))
joblib.dump(encoder, '/tmp/encoder.pkl')
print('encoder size:', os.path.getsize('/tmp/encoder.pkl'), 'bytes')
"
```

Expect non-zero counts everywhere, the documented feature columns, and
an encoder pickle of a few hundred KB to a few MB.

#### 4. Leakage / determinism guards fired

```bash
uv run pytest tests/test_encoder.py::test_encoder_is_deterministic_in_train -v
uv run pytest tests/test_encoder.py::test_encoder_pickle_round_trip -v
```

If either fails, the encoder has either started reading test data or
its dict construction lost determinism — fix before merging anything else.

### Not yet testable at this checkpoint

- LightGBM ranker training + offline metrics (Recall@K, NDCG@K) — checkpoint 3
- Co-listen / track-track similarity feature — deferred to v2 checkpoint
- API serving, blue/green load balancer, Grafana, UI — later checkpoints

---

<!-- Future checkpoints get appended below as they land:
## Checkpoint 3 — LightGBM ranker + MLflow training pipeline
## Checkpoint 4 — FastAPI serving layer
## Checkpoint 5 — Python load balancer + docker-compose
## Checkpoint 6 — Event replayer + online evaluation + Grafana
## Checkpoint 7 — Streamlit UI
-->
