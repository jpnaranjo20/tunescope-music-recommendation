"""Tests for the feature encoder, label builder, time-based split, and
negative sampler. Fixtures use a small deterministic synthetic dataset
so tests stay fast (~1s each).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_synthetic import GenConfig, generate
from tunescope.data.loader import load
from tunescope.features.encode import (
    Encoder,
    build_labels,
    fit_encoder,
    split_by_time,
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
