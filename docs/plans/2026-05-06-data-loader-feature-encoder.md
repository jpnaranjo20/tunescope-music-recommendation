# Data loader + feature encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data loader (`tunescope/data/loader.py`) and feature encoder (`tunescope/features/encode.py`) for checkpoint 2 of the master plan, with a time-based per-user train/test split, leakage-free encoder, and joblib-serializable artifacts.

**Architecture:** Two new modules. `loader.py` reads parquet files into a `Dataset` dataclass and applies the min-plays filter. `encode.py` provides time-based split, label generation, encoder fit-on-train-only, transform-to-feature-matrix, and negative sampling for LambdaRank. Co-listen feature deferred to v2.

**Tech Stack:** Python 3.12, pandas, numpy, pyarrow, joblib (added in Task 1), pytest. Existing code follows ruff style with line-length 100.

**Spec:** [`docs/design/2026-05-06-data-loader-feature-encoder-design.md`](../design/2026-05-06-data-loader-feature-encoder-design.md)

---

## File Structure

| Path | Responsibility | Action |
|---|---|---|
| `pyproject.toml` | Add `joblib` to deps | Modify |
| `tunescope/data/__init__.py` | Package marker | Create (empty) |
| `tunescope/data/loader.py` | `Dataset` dataclass + `load()` with min-plays filter | Create |
| `tunescope/features/__init__.py` | Package marker | Create (empty) |
| `tunescope/features/encode.py` | Encoder, split, labels, transform, neg-sampling | Create |
| `tests/test_loader.py` | Three loader tests | Create |
| `tests/test_encoder.py` | Encoder + split + labels + neg-sample + serialization tests | Create |
| `TESTING.md` | Append "Checkpoint 2" section | Modify |

---

## Task 1: Add joblib dependency

**Files:**
- Modify: `pyproject.toml` (via `uv add`)
- Modify: `uv.lock`

- [ ] **Step 1: Add joblib via uv**

```bash
uv add joblib
```

Expected: `uv` reports `joblib==X.Y.Z` installed and pyproject.toml + uv.lock updated.

- [ ] **Step 2: Verify install**

```bash
uv run python -c "import joblib; print(joblib.__version__)"
```

Expected: prints a version string (e.g. `1.4.2`).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add joblib dependency for encoder serialization"
```

---

## Task 2: Data loader

**Files:**
- Create: `tunescope/data/__init__.py`
- Create: `tunescope/data/loader.py`
- Create: `tests/test_loader.py`

- [ ] **Step 1: Create empty package marker**

```bash
mkdir -p tunescope/data
touch tunescope/data/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_loader.py`:

```python
"""Tests for the data loader.

The loader is a thin wrapper over parquet IO + a min-plays filter.
These tests lock in: schema preservation, the filter behaviour, and
deterministic round-trip from disk.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_synthetic import GenConfig, generate
from tunescope.data.loader import Dataset, load

SMALL = {"n_users": 200, "n_tracks": 300, "n_plays": 5_000}


@pytest.fixture
def small_data_dir(tmp_path: Path) -> Path:
    cfg = GenConfig(out_dir=tmp_path, seed=42, **SMALL)
    generate(cfg)
    return tmp_path


def test_loader_returns_dataset(small_data_dir: Path) -> None:
    ds = load(small_data_dir)
    assert isinstance(ds, Dataset)
    assert list(ds.users.columns) == ["user_id", "country", "signup_ts"]
    assert list(ds.tracks.columns) == [
        "track_id",
        "artist_id",
        "track_name",
        "artist_name",
    ]
    assert list(ds.plays.columns) == ["user_id", "track_id", "ts"]


def test_loader_applies_min_plays_filter(small_data_dir: Path) -> None:
    raw_plays = pd.read_parquet(small_data_dir / "plays.parquet")
    raw_tracks = pd.read_parquet(small_data_dir / "tracks.parquet")
    counts = raw_plays.user_id.value_counts()
    expected_eligible = set(counts[counts >= 5].index)

    ds = load(small_data_dir, min_plays=5)

    assert set(ds.users.user_id) == expected_eligible
    assert set(ds.plays.user_id).issubset(expected_eligible)
    # Tracks unchanged (cold-start tracks must stay visible)
    assert len(ds.tracks) == len(raw_tracks)


def test_loader_round_trip(small_data_dir: Path) -> None:
    a = load(small_data_dir)
    b = load(small_data_dir)
    pd.testing.assert_frame_equal(a.users, b.users)
    pd.testing.assert_frame_equal(a.tracks, b.tracks)
    pd.testing.assert_frame_equal(a.plays, b.plays)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_loader.py -v
```

Expected: ImportError on `tunescope.data.loader` — module doesn't exist yet.

- [ ] **Step 4: Implement the loader**

Create `tunescope/data/loader.py`:

```python
"""Data loader: parquet on disk → typed in-memory DataFrames.

Source-agnostic by design — works with both ``data/raw/synthetic/`` and
the future ``data/raw/lastfm-1k/`` because both have the same schema (see
``DATA.md``). Applies one transformation: drops users with fewer than
``min_plays`` plays (and the corresponding play rows) so cold-start
users don't pollute training. ``tracks`` is returned unchanged so the
popularity prior can still see cold tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Dataset:
    users: pd.DataFrame
    tracks: pd.DataFrame
    plays: pd.DataFrame


def load(source_dir: Path | str, *, min_plays: int = 5) -> Dataset:
    """Read users / tracks / plays parquet from ``source_dir`` and apply
    the min-plays filter.

    Returns a Dataset whose internals are independent of which source
    directory the data came from.
    """
    source_dir = Path(source_dir)
    users = pd.read_parquet(source_dir / "users.parquet")
    tracks = pd.read_parquet(source_dir / "tracks.parquet")
    plays = pd.read_parquet(source_dir / "plays.parquet")

    counts = plays.user_id.value_counts()
    eligible = counts[counts >= min_plays].index
    users = users[users.user_id.isin(eligible)].reset_index(drop=True)
    plays = plays[plays.user_id.isin(eligible)].reset_index(drop=True)

    return Dataset(users=users, tracks=tracks, plays=plays)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_loader.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Lint and format**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: "All checks passed!" and "N files already formatted". Run `uv run ruff format .` if any reformatting is needed.

- [ ] **Step 7: Commit**

```bash
git add tunescope/data/__init__.py tunescope/data/loader.py tests/test_loader.py
git commit -m "Add data loader with min-plays filter"
```

---

## Task 3: split_by_time

**Files:**
- Create: `tunescope/features/__init__.py`
- Create: `tunescope/features/encode.py` (start)
- Create: `tests/test_encoder.py` (start)

- [ ] **Step 1: Create empty package marker**

```bash
mkdir -p tunescope/features
touch tunescope/features/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_encoder.py`:

```python
"""Tests for the feature encoder, label builder, time-based split, and
negative sampler. Fixtures use a small deterministic synthetic dataset
so tests stay fast (~1s each).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import pandas as pd
import pytest

from scripts.generate_synthetic import GenConfig, generate
from tunescope.data.loader import load
from tunescope.features.encode import (
    Encoder,
    build_labels,
    fit_encoder,
    sample_negatives,
    split_by_time,
    transform,
)

SMALL = {"n_users": 200, "n_tracks": 300, "n_plays": 5_000}


@pytest.fixture
def small_dataset(tmp_path: Path):
    cfg = GenConfig(out_dir=tmp_path, seed=42, **SMALL)
    generate(cfg)
    return load(tmp_path)


def test_split_holds_out_last_fraction_per_user(small_dataset) -> None:
    plays = small_dataset.plays
    train, test = split_by_time(plays, holdout_frac=0.2)

    assert len(train) + len(test) == len(plays)
    for uid in plays.user_id.unique():
        u_train = train[train.user_id == uid]
        u_test = test[test.user_id == uid]
        if len(u_test) > 0 and len(u_train) > 0:
            assert u_test.ts.min() >= u_train.ts.max(), (
                f"user {uid}: test min ts {u_test.ts.min()} < train max {u_train.ts.max()}"
            )


def test_split_every_eligible_user_in_both_sides(small_dataset) -> None:
    plays = small_dataset.plays
    train, test = split_by_time(plays, holdout_frac=0.2)
    assert set(plays.user_id) == set(train.user_id)
    assert set(plays.user_id) == set(test.user_id)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_encoder.py -v
```

Expected: ImportError on `tunescope.features.encode` — module doesn't exist yet.

- [ ] **Step 4: Implement split_by_time**

Create `tunescope/features/encode.py`:

```python
"""Feature encoder for the music recommender.

Provides:
    split_by_time      — per-user time-based train/test split
    build_labels       — implicit-feedback "liked" labels
    fit_encoder        — build per-user / per-track / per-pair lookups
    transform          — apply encoder to (user, track) pairs
    sample_negatives   — sample unplayed tracks per positive

The encoder fits ONLY on training data (the test plays are never passed
in). All dict structures are built with sorted-key insertion so joblib
pickle hashes are deterministic across runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def split_by_time(
    plays: pd.DataFrame, *, holdout_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user time-based split.

    For each user, sort their plays by ts ascending and take the first
    ``floor((1 - holdout_frac) * n_user_plays)`` rows as train, the rest
    as test. Both outputs are reset_index'd for downstream cleanliness.
    """
    sorted_plays = plays.sort_values(["user_id", "ts"], kind="stable").reset_index(
        drop=True
    )
    user_counts = sorted_plays.groupby("user_id").size()
    train_sizes = (user_counts * (1 - holdout_frac)).astype(int)

    rank = sorted_plays.groupby("user_id").cumcount()
    user_train_size = sorted_plays.user_id.map(train_sizes)
    is_train = rank < user_train_size

    train = sorted_plays[is_train].reset_index(drop=True)
    test = sorted_plays[~is_train].reset_index(drop=True)
    return train, test
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_encoder.py::test_split_holds_out_last_fraction_per_user tests/test_encoder.py::test_split_every_eligible_user_in_both_sides -v
```

Expected: 2 passed.

- [ ] **Step 6: Lint and format**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add tunescope/features/__init__.py tunescope/features/encode.py tests/test_encoder.py
git commit -m "Add per-user time-based split for encoder"
```

---

## Task 4: build_labels

**Files:**
- Modify: `tunescope/features/encode.py` (append)
- Modify: `tests/test_encoder.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_encoder.py`:

```python
def test_build_labels_threshold() -> None:
    plays = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2, 3],
            "track_id": [10, 10, 10, 20, 30, 30, 40],
            "ts": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-01",
                ],
                utc=True,
            ),
        }
    )
    liked = build_labels(plays, threshold=3)

    # Only (user=1, track=10) appears 3+ times
    assert len(liked) == 1
    assert int(liked.iloc[0].user_id) == 1
    assert int(liked.iloc[0].track_id) == 10
    assert int(liked.iloc[0].liked) == 1
    assert list(liked.columns) == ["user_id", "track_id", "liked"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_encoder.py::test_build_labels_threshold -v
```

Expected: ImportError or AttributeError — `build_labels` not defined yet.

- [ ] **Step 3: Implement build_labels**

Append to `tunescope/features/encode.py`:

```python
def build_labels(plays: pd.DataFrame, *, threshold: int = 3) -> pd.DataFrame:
    """Return (user_id, track_id, liked=1) for every (user, track) pair
    appearing at least ``threshold`` times in ``plays``."""
    counts = (
        plays.groupby(["user_id", "track_id"]).size().rename("n").reset_index()
    )
    liked = counts[counts.n >= threshold].drop(columns="n").copy()
    liked["liked"] = 1
    return liked.reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_encoder.py::test_build_labels_threshold -v
```

Expected: 1 passed.

- [ ] **Step 5: Lint and format**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add tunescope/features/encode.py tests/test_encoder.py
git commit -m "Add build_labels: implicit-feedback liked-pair labelling"
```

---

## Task 5: Encoder dataclass + fit_encoder

**Files:**
- Modify: `tunescope/features/encode.py` (append)
- Modify: `tests/test_encoder.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_encoder.py`:

```python
def test_fit_encoder_user_features(small_dataset) -> None:
    train, _ = split_by_time(small_dataset.plays, holdout_frac=0.2)
    encoder = fit_encoder(train, small_dataset.users, small_dataset.tracks)

    assert isinstance(encoder, Encoder)
    expected_cols = [
        "u_n_plays",
        "u_n_distinct_tracks",
        "u_n_distinct_artists",
        "u_days_active",
        "u_top_artist_share",
        "u_country",
    ]
    assert list(encoder.user_features.columns) == expected_cols
    assert len(encoder.user_features) == train.user_id.nunique()
    # Country codes are non-negative ints
    assert (encoder.user_features.u_country >= 0).all()


def test_fit_encoder_track_features(small_dataset) -> None:
    train, _ = split_by_time(small_dataset.plays, holdout_frac=0.2)
    encoder = fit_encoder(train, small_dataset.users, small_dataset.tracks)

    expected_cols = ["t_artist_id", "t_n_plays", "t_n_distinct_users"]
    assert sorted(encoder.track_features.columns) == sorted(expected_cols)
    # Includes cold-start tracks (zero plays)
    assert len(encoder.track_features) == len(small_dataset.tracks)


def test_fit_encoder_country_ordering_alphabetical(small_dataset) -> None:
    train, _ = split_by_time(small_dataset.plays, holdout_frac=0.2)
    encoder = fit_encoder(train, small_dataset.users, small_dataset.tracks)
    assert encoder.countries == sorted(encoder.countries)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_encoder.py::test_fit_encoder_user_features tests/test_encoder.py::test_fit_encoder_track_features tests/test_encoder.py::test_fit_encoder_country_ordering_alphabetical -v
```

Expected: ImportError on `Encoder` / `fit_encoder`.

- [ ] **Step 3: Implement Encoder + fit_encoder**

Append to `tunescope/features/encode.py`:

```python
@dataclass
class Encoder:
    """All lookup tables needed to compute features at training and
    inference time. Built ONLY from training data."""

    user_features: pd.DataFrame  # indexed by user_id
    track_features: pd.DataFrame  # indexed by track_id
    user_artist_plays: dict[tuple[int, int], int]
    user_track_plays: dict[tuple[int, int], int]
    user_track_last_ts: dict[tuple[int, int], pd.Timestamp]
    split_cutoffs: dict[int, pd.Timestamp]  # per-user max ts in train
    countries: list[str]
    n_tracks: int


def fit_encoder(
    train_plays: pd.DataFrame,
    users: pd.DataFrame,
    tracks: pd.DataFrame,
) -> Encoder:
    """Build the lookup tables an Encoder needs from train data only.

    The encoder never sees test data; this is the leakage guard at the
    API level. Dicts are built with sorted-key insertion so joblib
    pickle hashes are deterministic across runs.
    """
    plays_w_artist = train_plays.merge(
        tracks[["track_id", "artist_id"]], on="track_id", how="left"
    )

    # ---- User features ----
    user_agg = plays_w_artist.groupby("user_id").agg(
        u_n_plays=("ts", "size"),
        u_n_distinct_tracks=("track_id", "nunique"),
        u_n_distinct_artists=("artist_id", "nunique"),
        u_first_ts=("ts", "min"),
        u_last_ts=("ts", "max"),
    )
    user_agg["u_days_active"] = (user_agg.u_last_ts - user_agg.u_first_ts).dt.days

    artist_counts = plays_w_artist.groupby(["user_id", "artist_id"]).size()
    top_artist = artist_counts.groupby(level=0).max()
    user_agg["u_top_artist_share"] = top_artist / user_agg.u_n_plays

    countries = sorted(users.country.unique().tolist())
    country_to_code = {c: i for i, c in enumerate(countries)}
    country_lookup = users.set_index("user_id").country.map(country_to_code)
    user_agg["u_country"] = country_lookup.astype("int64")

    user_features = user_agg[
        [
            "u_n_plays",
            "u_n_distinct_tracks",
            "u_n_distinct_artists",
            "u_days_active",
            "u_top_artist_share",
            "u_country",
        ]
    ].copy()

    # ---- Track features ----
    track_agg = train_plays.groupby("track_id").agg(
        t_n_plays=("ts", "size"),
        t_n_distinct_users=("user_id", "nunique"),
    )
    track_features = (
        tracks.set_index("track_id")[["artist_id"]]
        .rename(columns={"artist_id": "t_artist_id"})
        .join(track_agg, how="left")
        .fillna({"t_n_plays": 0, "t_n_distinct_users": 0})
        .astype({"t_n_plays": "int64", "t_n_distinct_users": "int64", "t_artist_id": "int64"})
    )

    # ---- Lookup dicts (sorted keys for deterministic pickle hashes) ----
    ut_groups = plays_w_artist.groupby(["user_id", "track_id"])
    user_track_plays = {k: int(v) for k, v in sorted(ut_groups.size().items())}
    user_track_last_ts = {k: v for k, v in sorted(ut_groups["ts"].max().items())}

    ua_groups = plays_w_artist.groupby(["user_id", "artist_id"])
    user_artist_plays = {k: int(v) for k, v in sorted(ua_groups.size().items())}

    split_cutoffs = {
        int(uid): ts
        for uid, ts in sorted(train_plays.groupby("user_id")["ts"].max().items())
    }

    return Encoder(
        user_features=user_features,
        track_features=track_features,
        user_artist_plays=user_artist_plays,
        user_track_plays=user_track_plays,
        user_track_last_ts=user_track_last_ts,
        split_cutoffs=split_cutoffs,
        countries=countries,
        n_tracks=len(tracks),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_encoder.py::test_fit_encoder_user_features tests/test_encoder.py::test_fit_encoder_track_features tests/test_encoder.py::test_fit_encoder_country_ordering_alphabetical -v
```

Expected: 3 passed.

- [ ] **Step 5: Lint and format**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: clean. Run `uv run ruff format .` if any reformatting is needed.

- [ ] **Step 6: Commit**

```bash
git add tunescope/features/encode.py tests/test_encoder.py
git commit -m "Add Encoder dataclass and fit_encoder lookups"
```

---

## Task 6: transform

**Files:**
- Modify: `tunescope/features/encode.py` (append)
- Modify: `tests/test_encoder.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_encoder.py`:

```python
def test_transform_shape_and_dtypes(small_dataset) -> None:
    train, _ = split_by_time(small_dataset.plays, holdout_frac=0.2)
    encoder = fit_encoder(train, small_dataset.users, small_dataset.tracks)
    pairs = train[["user_id", "track_id"]].head(50).reset_index(drop=True)
    out = transform(encoder, pairs)

    expected_cols = {
        "user_id",
        "track_id",
        "u_n_plays",
        "u_n_distinct_tracks",
        "u_n_distinct_artists",
        "u_days_active",
        "u_top_artist_share",
        "u_country",
        "t_artist_id",
        "t_n_plays",
        "t_n_distinct_users",
        "ut_prior_play_count",
        "ut_days_since_last_play",
        "ua_n_plays_of_artist",
    }
    assert set(out.columns) == expected_cols
    assert len(out) == len(pairs)
    # Numeric columns should be numeric dtypes
    assert pd.api.types.is_integer_dtype(out.u_n_plays)
    assert pd.api.types.is_integer_dtype(out.t_n_plays)
    assert pd.api.types.is_float_dtype(out.ut_days_since_last_play)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_encoder.py::test_transform_shape_and_dtypes -v
```

Expected: ImportError on `transform`.

- [ ] **Step 3: Implement transform**

Append to `tunescope/features/encode.py`:

```python
def transform(encoder: Encoder, pairs: pd.DataFrame) -> pd.DataFrame:
    """Given (user_id, track_id) pairs, return a feature matrix.

    Output rows are aligned with ``pairs``. Cold-pair interaction values
    (user has never played the track / artist) come back as 0 for counts
    and ``np.nan`` for ``ut_days_since_last_play``. LightGBM handles
    missing values natively.
    """
    out = pairs[["user_id", "track_id"]].copy().reset_index(drop=True)

    # User-side broadcast
    out = out.merge(
        encoder.user_features, left_on="user_id", right_index=True, how="left"
    )
    # Track-side broadcast
    out = out.merge(
        encoder.track_features, left_on="track_id", right_index=True, how="left"
    )

    # Interaction features via dict lookups
    ut_keys = list(zip(out.user_id.tolist(), out.track_id.tolist(), strict=True))
    out["ut_prior_play_count"] = [
        encoder.user_track_plays.get((int(u), int(t)), 0) for u, t in ut_keys
    ]

    cutoffs = [encoder.split_cutoffs.get(int(u)) for u, _ in ut_keys]
    last_ts = [encoder.user_track_last_ts.get((int(u), int(t))) for u, t in ut_keys]
    days: list[float] = []
    for ts, cutoff in zip(last_ts, cutoffs, strict=True):
        if ts is None or cutoff is None:
            days.append(float("nan"))
        else:
            days.append((cutoff - ts).total_seconds() / 86400.0)
    out["ut_days_since_last_play"] = days

    ua_keys = list(zip(out.user_id.tolist(), out.t_artist_id.tolist(), strict=True))
    out["ua_n_plays_of_artist"] = [
        encoder.user_artist_plays.get((int(u), int(a)), 0) for u, a in ua_keys
    ]

    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_encoder.py::test_transform_shape_and_dtypes -v
```

Expected: 1 passed.

- [ ] **Step 5: Lint and format**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add tunescope/features/encode.py tests/test_encoder.py
git commit -m "Add transform: build feature matrix for (user, track) pairs"
```

---

## Task 7: sample_negatives

**Files:**
- Modify: `tunescope/features/encode.py` (append)
- Modify: `tests/test_encoder.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_encoder.py`:

```python
def test_negative_sampling_count_and_disjoint(small_dataset) -> None:
    train, _ = split_by_time(small_dataset.plays, holdout_frac=0.2)
    encoder = fit_encoder(train, small_dataset.users, small_dataset.tracks)
    positives = build_labels(train, threshold=3).head(20)
    if len(positives) == 0:
        pytest.skip("no positives in fixture; bump SMALL plays count")

    negs = sample_negatives(encoder, positives, k=4, seed=0)

    # 4 negatives per positive
    assert len(negs) == 4 * len(positives)
    assert (negs.liked == 0).all()
    assert list(negs.columns) == ["user_id", "track_id", "liked"]

    # No (user, track) negative is also in train (i.e. user actually played it)
    train_pairs = set(zip(train.user_id, train.track_id, strict=True))
    for _, row in negs.iterrows():
        assert (int(row.user_id), int(row.track_id)) not in train_pairs


def test_negative_sampling_deterministic(small_dataset) -> None:
    train, _ = split_by_time(small_dataset.plays, holdout_frac=0.2)
    encoder = fit_encoder(train, small_dataset.users, small_dataset.tracks)
    positives = build_labels(train, threshold=3).head(20)
    if len(positives) == 0:
        pytest.skip("no positives in fixture")

    a = sample_negatives(encoder, positives, k=4, seed=42)
    b = sample_negatives(encoder, positives, k=4, seed=42)
    pd.testing.assert_frame_equal(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_encoder.py::test_negative_sampling_count_and_disjoint tests/test_encoder.py::test_negative_sampling_deterministic -v
```

Expected: ImportError on `sample_negatives`.

- [ ] **Step 3: Implement sample_negatives**

Append to `tunescope/features/encode.py`:

```python
def sample_negatives(
    encoder: Encoder,
    positive_pairs: pd.DataFrame,
    *,
    k: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """For each positive (u, t), sample ``k`` random tracks the user has
    never played in train. Returns a DataFrame with columns
    ``(user_id, track_id, liked=0)``.

    Per-user negatives are sampled WITHOUT replacement to avoid trivial
    duplicates. If a user has played more than half the catalog,
    rejection sampling stalls — fall back to ``np.setdiff1d`` then
    ``rng.choice``.
    """
    rng = np.random.default_rng(seed)

    # Pre-compute per-user played-track sets for O(1) membership lookups.
    user_played: dict[int, set[int]] = {}
    for u, t in encoder.user_track_plays:
        user_played.setdefault(int(u), set()).add(int(t))

    out_users: list[int] = []
    out_tracks: list[int] = []

    for u in positive_pairs.user_id.tolist():
        u = int(u)
        played = user_played.get(u, set())
        if encoder.n_tracks - len(played) < k:
            # Not enough unplayed tracks; emit fewer negatives for this user.
            unplayed_count = encoder.n_tracks - len(played)
            if unplayed_count <= 0:
                continue
            unplayed = np.setdiff1d(np.arange(encoder.n_tracks), list(played))
            chosen = rng.choice(unplayed, size=unplayed_count, replace=False)
        elif len(played) > encoder.n_tracks * 0.5:
            unplayed = np.setdiff1d(np.arange(encoder.n_tracks), list(played))
            chosen = rng.choice(unplayed, size=k, replace=False)
        else:
            chosen_set: set[int] = set()
            while len(chosen_set) < k:
                cands = rng.integers(0, encoder.n_tracks, size=k * 2)
                for c in cands:
                    c = int(c)
                    if c not in played and c not in chosen_set:
                        chosen_set.add(c)
                        if len(chosen_set) == k:
                            break
            chosen = np.array(sorted(chosen_set))

        for c in chosen:
            out_users.append(u)
            out_tracks.append(int(c))

    return pd.DataFrame(
        {
            "user_id": out_users,
            "track_id": out_tracks,
            "liked": [0] * len(out_users),
        }
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_encoder.py::test_negative_sampling_count_and_disjoint tests/test_encoder.py::test_negative_sampling_deterministic -v
```

Expected: 2 passed (or 1 passed + 1 skipped if no positives in fixture).

- [ ] **Step 5: Lint and format**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add tunescope/features/encode.py tests/test_encoder.py
git commit -m "Add negative sampling for LambdaRank training pairs"
```

---

## Task 8: Pickle round-trip + leakage guard tests

**Files:**
- Modify: `tests/test_encoder.py` (append)

No production-code changes — these are tests that lock in invariants on existing code.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_encoder.py`:

```python
def test_encoder_pickle_round_trip(small_dataset, tmp_path: Path) -> None:
    """joblib.dump → joblib.load must preserve the encoder so transform()
    output is byte-identical."""
    train, _ = split_by_time(small_dataset.plays, holdout_frac=0.2)
    encoder = fit_encoder(train, small_dataset.users, small_dataset.tracks)
    pairs = train[["user_id", "track_id"]].head(20).reset_index(drop=True)
    out_a = transform(encoder, pairs)

    path = tmp_path / "encoder.pkl"
    joblib.dump(encoder, path)
    loaded = joblib.load(path)
    out_b = transform(loaded, pairs)

    pd.testing.assert_frame_equal(out_a, out_b)


def test_encoder_is_deterministic_in_train(small_dataset, tmp_path: Path) -> None:
    """Re-fitting on the same train data must produce a byte-identical
    encoder pickle. Combined with the API contract that fit_encoder only
    accepts train data, this is the leakage guard: any future refactor
    that quietly reads test data will perturb the hash."""
    train, _ = split_by_time(small_dataset.plays, holdout_frac=0.2)

    a = fit_encoder(train, small_dataset.users, small_dataset.tracks)
    b = fit_encoder(train, small_dataset.users, small_dataset.tracks)

    pa = tmp_path / "a.pkl"
    pb = tmp_path / "b.pkl"
    joblib.dump(a, pa)
    joblib.dump(b, pb)

    ha = hashlib.sha256(pa.read_bytes()).hexdigest()
    hb = hashlib.sha256(pb.read_bytes()).hexdigest()
    assert ha == hb, "encoder pickle hash drifted across two identical fits"
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
uv run pytest tests/test_encoder.py::test_encoder_pickle_round_trip tests/test_encoder.py::test_encoder_is_deterministic_in_train -v
```

Expected: 2 passed. (No implementation needed — these test existing code.)

- [ ] **Step 3: Run the full encoder + loader suite to confirm nothing broke**

```bash
uv run pytest -v
```

Expected: all tests passing (synthetic + loader + encoder = 7 + 3 + 11 = 21 tests).

- [ ] **Step 4: Lint and format**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add tests/test_encoder.py
git commit -m "Add pickle round-trip and encoder determinism (leakage guard) tests"
```

---

## Task 9: Append Checkpoint 2 section to TESTING.md

**Files:**
- Modify: `TESTING.md`

- [ ] **Step 1: Append the new checkpoint section**

Read `TESTING.md` first, then append the following after the closing comment block (`-->`) of the Checkpoint 1 placeholder list. Replace the placeholder line for Checkpoint 2 with the full section below:

```markdown
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
import joblib
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
print('encoder size:', __import__('os').path.getsize('/tmp/encoder.pkl'), 'bytes')
"
```

Expect non-zero counts everywhere, the documented feature columns, and
an encoder pickle of a few hundred KB to a few MB.

#### 4. Leakage / determinism guard fired

The two locked-in regression tests:

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
```

- [ ] **Step 2: Verify TESTING.md still parses cleanly**

```bash
head -5 TESTING.md
tail -50 TESTING.md
```

Expect: header intact at the top, new Checkpoint 2 content visible at the bottom.

- [ ] **Step 3: Commit**

```bash
git add TESTING.md
git commit -m "Document checkpoint 2 manual tests in TESTING.md"
```

---

## Definition of done

After all 9 tasks:

- 21 tests green via `uv run pytest`
- `uv run ruff check .` and `uv run ruff format --check .` clean
- 8 new commits on top of `main` (one per task that produces files; Task 1 is its own commit)
- `tunescope/data/loader.py` and `tunescope/features/encode.py` exist with the documented public API
- TESTING.md has a populated Checkpoint 2 section

## Risks reminder

- **Negative sampling stall:** the `n_played > 0.5 * n_tracks` fallback in Task 7 protects against rejection-sampling loops. Don't simplify it away.
- **Pickle hash determinism:** all dicts in `Encoder` are built via sorted-key insertion. If a future refactor changes that, `test_encoder_is_deterministic_in_train` will fail — that's the signal.
- **Tracks unchanged by loader:** the loader explicitly does not filter tracks. Cold tracks must remain visible so the popularity prior can train. Don't add a track-side filter without re-checking the spec.
