"""Generate a deterministic synthetic listening dataset.

Used as a fallback when the Last.fm 1K Users dataset is not available locally
(e.g. in CI, or when a developer wants to clone-and-run without the Kaggle
download). The shape mirrors what the real dataset will provide downstream:
users, tracks, and timestamped play events, with power-law distributions
over user activity and track popularity so the ranking model has realistic
implicit-feedback signal.

Output: three parquet files in ``data/raw/synthetic/``:

    users.parquet   user_id, country, signup_ts
    tracks.parquet  track_id, artist_id, track_name, artist_name
    plays.parquet   user_id, track_id, ts        (sorted by user_id, ts)

Run::

    uv run python scripts/generate_synthetic.py
    uv run python scripts/generate_synthetic.py --users 1000 --plays 50000 --seed 7
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_USERS = 5_000
DEFAULT_TRACKS = 5_000
DEFAULT_PLAYS = 500_000
DEFAULT_SEED = 1337
DEFAULT_OUT = Path("data/raw/synthetic")

# Listening events span this window; chosen to look plausible without
# mirroring the Last.fm 1K window exactly.
WINDOW_START = pd.Timestamp("2022-01-01", tz="UTC")
WINDOW_END = pd.Timestamp("2024-01-01", tz="UTC")

COUNTRIES = ("US", "GB", "DE", "BR", "JP", "CA", "FR", "AU", "MX", "ES")


@dataclass
class GenConfig:
    n_users: int
    n_tracks: int
    n_plays: int
    seed: int
    out_dir: Path


def _powerlaw_choice(
    rng: np.random.Generator, n_items: int, n_draws: int, alpha: float
) -> np.ndarray:
    """Sample ``n_draws`` item indices in [0, n_items) with a Zipf-like bias.

    Items with smaller indices are more popular. ``alpha`` controls steepness:
    higher values concentrate mass on the head.
    """
    # Weight item i by 1/(i+1)**alpha.
    weights = 1.0 / np.power(np.arange(1, n_items + 1, dtype=np.float64), alpha)
    weights /= weights.sum()
    return rng.choice(n_items, size=n_draws, replace=True, p=weights)


def _build_users(rng: np.random.Generator, n: int) -> pd.DataFrame:
    countries = rng.choice(COUNTRIES, size=n, replace=True)
    # Signups uniformly distributed over the year before the listening window.
    signup_window_ns = (WINDOW_START - pd.Timedelta(days=365)).value
    span_ns = WINDOW_START.value - signup_window_ns
    signup_ns = signup_window_ns + rng.integers(0, span_ns, size=n, dtype=np.int64)
    return pd.DataFrame(
        {
            "user_id": np.arange(n, dtype=np.int64),
            "country": countries,
            "signup_ts": pd.to_datetime(signup_ns, utc=True),
        }
    )


def _build_tracks(rng: np.random.Generator, n: int) -> pd.DataFrame:
    # ~1 artist per 5 tracks on average.
    n_artists = max(1, n // 5)
    artist_ids = rng.integers(0, n_artists, size=n, dtype=np.int64)
    return pd.DataFrame(
        {
            "track_id": np.arange(n, dtype=np.int64),
            "artist_id": artist_ids,
            "track_name": [f"track_{i}" for i in range(n)],
            "artist_name": [f"artist_{a}" for a in artist_ids],
        }
    )


def _build_plays(
    rng: np.random.Generator,
    n_users: int,
    n_tracks: int,
    n_plays: int,
) -> pd.DataFrame:
    # User activity is power-law: a small head listens a lot.
    user_ids = _powerlaw_choice(rng, n_users, n_plays, alpha=1.1)
    track_ids = _powerlaw_choice(rng, n_tracks, n_plays, alpha=1.0)

    # Timestamps: independent uniform draws over the window, then sorted
    # within each user so per-user event order is monotonic.
    span_ns = WINDOW_END.value - WINDOW_START.value
    ts_ns = WINDOW_START.value + rng.integers(0, span_ns, size=n_plays, dtype=np.int64)

    df = pd.DataFrame({"user_id": user_ids, "track_id": track_ids, "_ts_ns": ts_ns})
    df = df.sort_values(["user_id", "_ts_ns"], kind="stable").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df.pop("_ts_ns"), utc=True)
    return df


def generate(cfg: GenConfig) -> dict[str, Path]:
    """Generate the three parquet files and return their paths."""
    rng = np.random.default_rng(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    users = _build_users(rng, cfg.n_users)
    tracks = _build_tracks(rng, cfg.n_tracks)
    plays = _build_plays(rng, cfg.n_users, cfg.n_tracks, cfg.n_plays)

    paths = {
        "users": cfg.out_dir / "users.parquet",
        "tracks": cfg.out_dir / "tracks.parquet",
        "plays": cfg.out_dir / "plays.parquet",
    }
    users.to_parquet(paths["users"], index=False)
    tracks.to_parquet(paths["tracks"], index=False)
    plays.to_parquet(paths["plays"], index=False)
    return paths


def _parse_args(argv: list[str] | None = None) -> GenConfig:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--users", type=int, default=DEFAULT_USERS)
    p.add_argument("--tracks", type=int, default=DEFAULT_TRACKS)
    p.add_argument("--plays", type=int, default=DEFAULT_PLAYS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    return GenConfig(
        n_users=args.users,
        n_tracks=args.tracks,
        n_plays=args.plays,
        seed=args.seed,
        out_dir=args.out_dir,
    )


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    paths = generate(cfg)
    print(f"Wrote {cfg.n_users} users, {cfg.n_tracks} tracks, {cfg.n_plays} plays")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
