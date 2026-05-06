"""Tests for the synthetic dataset generator.

The downstream training pipeline trusts these invariants, so we lock them
in here:

* deterministic given a seed (CI must produce byte-identical artifacts run
  to run, otherwise model training reproducibility breaks);
* schema is exactly what the loader expects;
* per-user play timestamps are non-decreasing (event-replayer needs this);
* every user_id / track_id in plays exists in users / tracks (no orphans).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_synthetic import GenConfig, generate

# Small config keeps the suite fast; invariants don't depend on size.
SMALL = dict(n_users=200, n_tracks=300, n_plays=5_000)


def _make_cfg(tmp_path: Path, seed: int = 42) -> GenConfig:
    return GenConfig(out_dir=tmp_path, seed=seed, **SMALL)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def generated(tmp_path: Path) -> dict[str, pd.DataFrame]:
    paths = generate(_make_cfg(tmp_path))
    return {name: pd.read_parquet(p) for name, p in paths.items()}


def test_row_counts_match_config(generated: dict[str, pd.DataFrame]) -> None:
    assert len(generated["users"]) == SMALL["n_users"]
    assert len(generated["tracks"]) == SMALL["n_tracks"]
    assert len(generated["plays"]) == SMALL["n_plays"]


def test_schemas(generated: dict[str, pd.DataFrame]) -> None:
    assert list(generated["users"].columns) == ["user_id", "country", "signup_ts"]
    assert list(generated["tracks"].columns) == [
        "track_id",
        "artist_id",
        "track_name",
        "artist_name",
    ]
    assert list(generated["plays"].columns) == ["user_id", "track_id", "ts"]


def test_no_orphan_ids(generated: dict[str, pd.DataFrame]) -> None:
    user_ids = set(generated["users"]["user_id"])
    track_ids = set(generated["tracks"]["track_id"])
    plays = generated["plays"]
    assert set(plays["user_id"]).issubset(user_ids)
    assert set(plays["track_id"]).issubset(track_ids)


def test_per_user_timestamps_monotonic(generated: dict[str, pd.DataFrame]) -> None:
    plays = generated["plays"]
    # Within each user, ts must be non-decreasing — the event-replayer
    # streams events in this order.
    diffs = plays.groupby("user_id")["ts"].diff().dropna()
    assert (diffs >= pd.Timedelta(0)).all()


def test_deterministic_given_seed(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    paths_a = generate(GenConfig(out_dir=a, seed=42, **SMALL))
    paths_b = generate(GenConfig(out_dir=b, seed=42, **SMALL))
    for key in paths_a:
        assert _file_digest(paths_a[key]) == _file_digest(paths_b[key]), (
            f"{key}.parquet differs between two seeded runs"
        )


def test_different_seed_produces_different_data(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    paths_a = generate(GenConfig(out_dir=a, seed=1, **SMALL))
    paths_b = generate(GenConfig(out_dir=b, seed=2, **SMALL))
    # Sanity check: different seeds must not yield identical plays.
    assert _file_digest(paths_a["plays"]) != _file_digest(paths_b["plays"])
