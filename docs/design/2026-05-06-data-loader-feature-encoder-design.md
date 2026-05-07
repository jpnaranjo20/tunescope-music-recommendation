# Data loader + feature encoder — design

**Checkpoint:** 2 of the master plan in `~/.claude/plans/music-recommendation-service.md`.
**Status:** approved 2026-05-06. Implementation plan to follow.

## Context

The synthetic data generator (checkpoint 1) produces parquet files matching
the schema in `DATA.md`. Before we can train a LightGBM ranker, we need:

1. A loader that reads those parquet files into typed in-memory DataFrames,
   independent of whether the source is the synthetic generator or the real
   Last.fm 1K dataset.
2. A feature encoder that turns those DataFrames into the `(user, track,
   features, label)` rows LightGBM trains on, with a clean train/test split
   and zero leakage from test data into the encoder.

Both modules ship in this checkpoint. The model itself is checkpoint 3.

### Decisions inherited from prior conversations

| decision | source |
|---|---|
| Min-plays filter for training: drop users with `<5` plays | EDA findings (cold-start cohort) |
| Label rule: `liked = (user, track)` with `>= 3` plays in train window | EDA findings |
| Eval cohort: users with `>= 10` liked tracks (~250 in synthetic) | EDA findings |
| Storage hybrid: parquet for offline corpus, SQLite for serving events | project memory, 2026-05-06 |
| Co-listen feature deferred to v2 | this brainstorm |

## Scope

### In scope (this checkpoint)

- `tunescope/data/loader.py` — read parquet, apply min-plays filter, return
  a `Dataset` dataclass.
- `tunescope/features/encode.py` — time-based per-user train/test split,
  label generation, encoder fit on train only, transform to feature matrix,
  negative sampling for LambdaRank, joblib serialization.
- Tests for both modules (10 cases listed below), all running on the
  deterministic synthetic dataset.

### Out of scope (deferred)

- **Co-listen / track-track similarity matrix.** Will be added in a v2
  checkpoint after the baseline pipeline is end-to-end. The encoder API is
  designed so this slots in as one extra interaction feature without
  reshaping anything.
- **Time-of-day / day-of-week features.** No signal on synthetic data
  (sessions are placed at uniformly-random calendar moments); revisit when
  Last.fm 1K is wired up.
- **Hyperparameter sweeps over feature engineering choices.** One reasonable
  set of features is enough.
- **Materializing feature matrices to parquet on disk.** Train and test
  matrices flow directly from `transform()` into LightGBM in checkpoint 3.

## Components

### `tunescope/data/loader.py`

```python
@dataclass
class Dataset:
    users: pd.DataFrame   # user_id, country, signup_ts
    tracks: pd.DataFrame  # track_id, artist_id, track_name, artist_name
    plays: pd.DataFrame   # user_id, track_id, ts


def load(source_dir: Path, *, min_plays: int = 5) -> Dataset:
    """Read users / tracks / plays parquet from source_dir.

    Drops users with fewer than `min_plays` plays and the corresponding
    play rows. `tracks` is returned unchanged (cold-start tracks stay
    visible so the model can learn popularity-prior features).

    Returns a Dataset whose internals are independent of whether
    source_dir contains synthetic or Last.fm 1K data.
    """
```

The loader does not know about feature engineering, splits, or labels.
Single responsibility: parquet → typed in-memory DataFrames with a
documented filter applied.

### `tunescope/features/encode.py`

```python
@dataclass
class Encoder:
    user_features: pd.DataFrame                  # indexed by user_id
    track_features: pd.DataFrame                 # indexed by track_id
    user_artist_plays: dict[tuple[int, int], int]  # (user_id, artist_id) -> count
    user_track_plays: dict[tuple[int, int], int]   # (user_id, track_id) -> count
    user_track_last_ts: dict[tuple[int, int], pd.Timestamp]
    split_cutoffs: dict[int, pd.Timestamp]       # per-user train/test boundary
    countries: list[str]                         # for stable categorical ordering


def split_by_time(
    plays: pd.DataFrame, *, holdout_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user time-based split. Returns (train_plays, test_plays)."""


def build_labels(
    plays: pd.DataFrame, *, threshold: int = 3
) -> pd.DataFrame:
    """Return DataFrame with columns (user_id, track_id, liked=1) for
    every (user, track) with at least `threshold` plays in `plays`."""


def fit_encoder(
    train_plays: pd.DataFrame, users: pd.DataFrame, tracks: pd.DataFrame
) -> Encoder:
    """Build all the lookup tables the encoder needs from train data only."""


def transform(
    encoder: Encoder, pairs: pd.DataFrame
) -> pd.DataFrame:
    """Given (user_id, track_id) pairs, return the feature matrix.

    Output columns: u_n_plays, u_n_distinct_tracks, u_n_distinct_artists,
    u_days_active, u_top_artist_share, u_country, t_n_plays,
    t_n_distinct_users, t_artist_id, ut_prior_play_count,
    ut_days_since_last_play, ua_n_plays_of_artist.
    """


def sample_negatives(
    encoder: Encoder, positive_pairs: pd.DataFrame, *, k: int = 4, seed: int = 0
) -> pd.DataFrame:
    """For each positive (u, t) pair, sample k random tracks the user has
    NEVER played in train. Returns DataFrame with (user_id, track_id, liked=0)."""
```

## Feature catalog

### User features (one row per `user_id`)

| feature | type | meaning |
|---|---|---|
| `u_n_plays` | int | total play count in train |
| `u_n_distinct_tracks` | int | distinct tracks played |
| `u_n_distinct_artists` | int | distinct artists played |
| `u_days_active` | int | `(max(ts) - min(ts)).days` for the user |
| `u_top_artist_share` | float | plays of top artist / total plays |
| `u_country` | categorical (int code) | mapped via `Encoder.countries` (alphabetically sorted, fit on train) — stable ordering across runs |

### Track features (one row per `track_id`)

| feature | type | meaning |
|---|---|---|
| `t_n_plays` | int | total play count in train |
| `t_n_distinct_users` | int | distinct users who played it |
| `t_artist_id` | categorical (raw int) | passed through unchanged; LightGBM is told via `categorical_feature` that this column is categorical at training time |

### Interaction features (one row per `(user, track)` pair at transform time)

| feature | type | meaning |
|---|---|---|
| `ut_prior_play_count` | int | times this user played this track in train (0 if never) |
| `ut_days_since_last_play` | float | days since user's last play of this track (np.nan if never) |
| `ua_n_plays_of_artist` | int | times this user played this track's artist in train |

`np.nan` for "never played" is intentional — LightGBM handles missing values
natively and learns the right thing for cold pairs.

## Train/test split

For each user (who passed the min-plays filter):

1. Sort the user's plays by `ts` ascending.
2. Take `floor((1 - holdout_frac) * n_user_plays)` plays from the start as
   train; the remaining as test.
3. Record `cutoff_ts` = train's last `ts` for that user in
   `Encoder.split_cutoffs`. Used by recency features and as a documentation
   anchor for the API at inference time.

**Invariants** (locked by tests):

- Every user surviving the loader's `min_plays >= 5` filter ends up with
  `>= 4` plays in train and `>= 1` in test (since `floor(5 * 0.8) = 4`),
  so neither side is ever empty for a user that reaches the split.
- For every user `u`: `min(test_plays[user==u].ts) >= max(train_plays[user==u].ts)`
  — strictly per-user; the split makes no global-time guarantee.
- `len(train_plays) + len(test_plays) == len(plays)` (no rows lost).

## Negative sampling

For LambdaRank training. Per positive `(u, t, liked=1)`:

1. Look up the set of tracks user `u` played in train (`encoder.user_track_plays.keys()` filtered by `u`).
2. Sample `k = 4` track ids from `tracks.track_id` that are NOT in that set.
3. Emit `(u, t', liked=0)` for each.

Sampling uses a seeded `np.random.Generator` so training is reproducible.
Group key for LambdaRank is `user_id`.

## Serialization

| artifact | path | format | written by |
|---|---|---|---|
| `Encoder` | `artifacts/<color>/encoder.pkl` | joblib | `pipeline.py` (next checkpoint) |
| Feature matrices | not persisted in this checkpoint | — | — |
| Split cutoffs | inside the encoder | — | — |

`<color>` is `blue` or `green`, mounted into the API containers per the
master plan's blue/green deploy structure.

## Testing strategy

All tests live in `tests/test_loader.py` and `tests/test_encoder.py`. Both
files use `scripts.generate_synthetic.generate` with `SMALL` config and a
fixed seed for determinism — same pattern as `tests/test_synthetic.py`.

| test | asserts |
|---|---|
| `test_loader_returns_dataset` | output shapes / dtypes match the schema in `DATA.md` |
| `test_loader_applies_min_plays_filter` | users with `<5` plays dropped from `users`; their plays dropped from `plays`; `tracks` unchanged |
| `test_loader_round_trip` | calling `load` twice on the same `source_dir` returns equal DataFrames |
| `test_split_holds_out_last_fraction_per_user` | for every user `u`, `min(test[u].ts) >= max(train[u].ts)`; per-user sizes within `±1` row of `holdout_frac` |
| `test_split_every_eligible_user_in_both_sides` | for users with `>= 5` plays, neither train nor test is empty |
| `test_build_labels_threshold` | `(u, t)` appears with `liked=1` iff `>= 3` plays in input |
| `test_encoder_only_uses_train_data` | perturbing the *test* set leaves encoder byte-identical via joblib hash |
| `test_negative_sampling_disjoint` | for every sampled `(u, t')`, `(u, t')` is not in `encoder.user_track_plays` |
| `test_transform_shape_and_dtypes` | output has documented columns and types; categorical features encoded |
| `test_encoder_pickle_round_trip` | `joblib.dump`/`load` produces an encoder whose `transform` output is byte-identical |

### Leakage test, in detail

`test_encoder_only_uses_train_data` is the most important guard. Procedure:

1. Load synthetic data → split → fit encoder → compute encoder hash A.
2. Mutate the **test** plays in arbitrary ways (drop rows, shuffle).
3. Re-run `fit_encoder` on the same train data → compute encoder hash B.
4. Assert `A == B`. Any failure means the encoder is reading from test
   under the table — fix immediately.

## Definition of done

This checkpoint is complete when:

1. `uv run python -c "from tunescope.data.loader import load; ds = load('data/raw/synthetic'); print(ds.users.shape, ds.tracks.shape, ds.plays.shape)"` runs without error and reports filter-applied shapes.
2. All 10 tests above are green via `uv run pytest`.
3. `uv run ruff check .` and `uv run ruff format --check .` are clean.
4. A new section is appended to `TESTING.md` for "Checkpoint 2 — data loader + feature encoder".
5. The work is split into two commits: one for the loader, one for the encoder + tests.

## Risks / things to watch

- **Python loops over users in negative sampling can be slow.** Vectorize where possible; if a user has played 99% of the track corpus, the rejection sampler can stall — fall back to `np.setdiff1d` once that ratio crosses 0.5.
- **`pd.Timestamp` arithmetic in `ut_days_since_last_play` returns `pd.Timedelta`.** Convert to `float` days explicitly so LightGBM sees a numeric column.
- **The encoder pickle hash equality depends on dict insertion order.** Build all dicts via sorted keys to keep the leakage test deterministic.
- **The min-plays filter is applied in the loader, not the encoder.** Eval-cohort filtering (`>= 10` liked tracks) happens later in the eval pipeline (checkpoint 3). Don't conflate the two.
