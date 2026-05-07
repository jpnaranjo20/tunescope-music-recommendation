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
