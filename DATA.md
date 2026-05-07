# Data — what we feed the model

This file documents how training data is shaped, generated, and (when
available) replaced with the real Last.fm 1K dataset. Read this before
touching `tunescope/data/loader.py` or `tunescope/features/encode.py`.

## Schema

Everything downstream — feature encoder, ranker, evaluation — assumes
three parquet files with these columns. Both data sources (synthetic,
Last.fm 1K) land in the same shape so downstream code is source-agnostic.

### `users.parquet`

| column      | dtype                  | meaning                                        |
|-------------|------------------------|------------------------------------------------|
| `user_id`   | `int64`                | dense integer id, `0..n_users-1`               |
| `country`   | `object` (string)      | ISO-style 2-letter code (`US`, `JP`, …)        |
| `signup_ts` | `datetime64[ns, UTC]`  | when the user joined; precedes the play window |

### `tracks.parquet`

| column        | dtype             | meaning                                  |
|---------------|-------------------|------------------------------------------|
| `track_id`    | `int64`           | dense integer id, `0..n_tracks-1`        |
| `artist_id`   | `int64`           | foreign key to a sparser artist space    |
| `track_name`  | `object` (string) | display label                            |
| `artist_name` | `object` (string) | display label                            |

There is no `artists.parquet` — artist identity lives only as
`(artist_id, artist_name)` columns on tracks. That's intentional: the
co-listen / artist-features the ranker uses can be computed by group-by
on tracks at training time, and the API never serves "an artist" as a
first-class entity.

### `plays.parquet`

| column     | dtype                 | meaning                            |
|------------|-----------------------|------------------------------------|
| `user_id`  | `int64`               | foreign key to `users`             |
| `track_id` | `int64`               | foreign key to `tracks`            |
| `ts`       | `datetime64[ns, UTC]` | when the user played the track    |

`ts` is short for **timestamp** — the column convention is borrowed from
the original Last.fm 1K dataset and kept compact because it appears in
many group-by, diff, and filter expressions downstream.

The file is sorted by `(user_id, ts)` ascending so per-user iteration is
streaming-friendly (no re-sort needed for windowed features or for the
event-replayer service).

## What a "session" is, and why we care

A **listening session** is one continuous sitting where a user plays a
burst of tracks. In real life, somebody sits down, listens to music
while they work / commute / do dishes for ~30–90 minutes, stops, and
hours or days later sits down again. The bursts are sessions; the
silence between them is what makes the inter-play gap distribution
bimodal.

For one of our synthetic users with 100 plays you might see something like:

```
session 1: Tue 2023-04-04, 19:30–20:25  → 18 plays  (avg 3 min between)
…silence: 6 days…
session 2: Mon 2023-04-10, 08:15–08:48  →  9 plays
…silence: 2 days…
session 3: Wed 2023-04-12, 21:00–22:30  → 23 plays
…
```

Why we care:

- **Time-window features need it.** "Plays in the last 24 hours" is only
  informative if 0 is a common answer (i.e. the user does have idle
  days). Without sessions, every user is permanently lukewarm-active and
  the feature carries no signal.
- **The event-replayer demo needs it.** The replayer streams plays in
  timestamp order at a configurable speedup factor and pushes them at
  the load balancer. With sessions, Grafana shows realistic bursty
  traffic; without, it shows a featureless flat line.
- **The "liked" label needs it.** Our implicit-feedback rule is "≥3
  plays of the same track in the window = liked." Real listeners hit
  that threshold by replaying favorites within the same session;
  uniform draws scatter repeats too far apart for it to fire.

## Synthetic generator

`scripts/generate_synthetic.py` produces a deterministic dataset
matching the schema above. Run with `uv run python scripts/generate_synthetic.py`.
Output goes to `data/raw/synthetic/` (gitignored).

### Step 1 — assign plays to (user, track) via two power laws

Each play picks a user and a track independently using a Zipf-like
weighting: item `i`'s probability is `1 / (i+1)^alpha`, normalized.
`alpha=1.1` for users, `alpha=1.0` for tracks. The first user / track
gets ~10–16% of all plays, with a long thin tail — the rank-frequency
plot is a clean straight line on log-log axes, which matches what real
listening data actually looks like.

### Step 2 — cluster each user's plays into sessions

For every user with `k` plays:

1. Pick `n_sessions = ceil(k / 15)` random calendar moments uniformly
   over the 2-year window. (15 plays/session is the design average.)
2. For each session, draw a duration from `Exponential(mean = 1 hour)`.
   Most sessions are short, a few are long — same shape as real listening.
3. Each of that user's plays picks one of those sessions uniformly at
   random, then jitters its timestamp uniformly within the session window.

Result: per-user timelines that are bursty (sessions visible at the
right zoom), aggregate daily volume that looks roughly steady (LLN
across thousands of users with random session days), and a bimodal
inter-play-gap distribution per user (intra-session gaps in minutes,
inter-session gaps in days).

### Determinism

The generator takes a `--seed` (default `1337`). Same seed → byte-identical
parquet output, including the session placements. Two pytest cases
(`test_deterministic_given_seed`, `test_different_seed_produces_different_data`)
guard this so CI artifacts stay reproducible across runs and so the
session structure can't silently regress to uniform Poisson rain
(`test_per_user_timestamps_show_session_clustering`).

### Defaults (what `make data` produces with no args)

| | value |
|---|---|
| users | 5,000 |
| tracks | 5,000 |
| plays | 500,000 |
| time window | 2022-01-01 → 2024-01-01 UTC |
| countries | 10 (US/GB/DE/BR/JP/CA/FR/AU/MX/ES) |
| seed | 1337 |
| size on disk | ~10 MB across the three parquet files |

All of the above are CLI-overridable via `--users`, `--tracks`,
`--plays`, `--seed`, `--out-dir`.

## Connection to Last.fm 1K

The synthetic generator is a **fallback**, not the primary path. The
real dataset of record is **Last.fm 1K Users** (Òscar Celma's research
dataset, mirrored on Kaggle): ~19M listening events from 1,000 users
with real per-event timestamps from 2005–2009. The model talk-track in
the README says "trained on Last.fm 1K"; synthetic exists so anyone
(CI, a recruiter who doesn't want a Kaggle account, a dev on a slow
network) can clone-and-run without that download.

### Where the real data lands

`scripts/download.sh` (planned, not yet built) fetches the archive into
`data/raw/lastfm-1k/`. Both `data/raw/lastfm-1k/` and `data/raw/synthetic/`
are gitignored — we never commit dataset bytes.

### Schema mapping

The Last.fm 1K archive is two TSV files; the loader normalizes them to
the same parquet schema as the synthetic data:

| our column            | Last.fm 1K source                                         |
|-----------------------|-----------------------------------------------------------|
| `users.user_id`       | `userid` (`user_000001` …) → dense integer remap          |
| `users.country`       | `userid-profile.tsv` → `country`                          |
| `users.signup_ts`     | `userid-profile.tsv` → `signup`                           |
| `tracks.track_id`     | `(musicbrainz-track-id, track-name)` → dense integer remap |
| `tracks.artist_id`    | `(musicbrainz-artist-id, artist-name)` → dense integer remap |
| `tracks.track_name`   | `track-name`                                              |
| `tracks.artist_name`  | `artist-name`                                             |
| `plays.user_id`       | `userid` (remapped)                                       |
| `plays.track_id`      | track remap                                               |
| `plays.ts`            | `timestamp` (ISO-8601, parsed UTC)                        |

The Last.fm profile rows also contain `gender` and `age`, which we drop
— we don't want demographic features in this project.

### `make data` resolves the source

The Makefile target:

```make
make data    # picks data/raw/lastfm-1k/ if present, else regenerates synthetic
```

Downstream code calls `tunescope.data.loader.load_plays()` (planned),
which reads whichever source `make data` produced and returns the same
three DataFrames. **Nothing past the loader knows or cares which source
the data came from** — that's the whole point of normalizing schemas
upstream.

### Why both, and what to expect when you switch

| | synthetic | Last.fm 1K |
|---|---|---|
| Users | 5,000 | 1,000 |
| Plays | 500,000 | ~19,000,000 |
| Plays/user (median) | low (power-law tail) | ~10,000+ |
| Time window | 2 years (2022–2024) | ~5 years (2005–2009) |
| Sessions | synthetic, mean 1h | real, varied |
| Cold tracks | many (long tail) | few |
| Repeat-listen rate | 22.5% of (u,t) pairs ≥2 plays | much higher (real listeners replay favorites a lot) |
| Eval cohort with ≥10 liked tracks | ~250 users (5%) | hundreds (most users) |

The model trains and evaluates on whichever is loaded. Recall@10 /
NDCG@10 numbers in interview talk-tracks come from the Last.fm run; CI
runs and quick demos use synthetic.

## Storage shape (parquet vs SQLite)

Training data is **parquet on disk** (offline, columnar, what every ML
library reads natively). Runtime data — the events the load balancer
and event-replayer write at serving time, joined against recommendations
for online hit-rate evaluation — is **SQLite**. We deliberately don't
collapse those into one storage primitive: parquet is the right tool
for an offline training corpus, SQL is the right tool for serving-layer
state. See the README for the production-scaling note (Postgres + S3
in a real deployment).
