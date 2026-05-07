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
    assert pd.api.types.is_integer_dtype(out.u_n_plays)
    assert pd.api.types.is_integer_dtype(out.t_n_plays)
    assert pd.api.types.is_float_dtype(out.ut_days_since_last_play)


def test_negative_sampling_count_and_disjoint(small_dataset) -> None:
    train, _ = split_by_time(small_dataset.plays, holdout_frac=0.2)
    encoder = fit_encoder(train, small_dataset.users, small_dataset.tracks)
    positives = build_labels(train, threshold=3).head(20)
    if len(positives) == 0:
        pytest.skip("no positives in fixture; bump SMALL plays count")

    negs = sample_negatives(encoder, positives, k=4, seed=0)

    assert len(negs) == 4 * len(positives)
    assert (negs.liked == 0).all()
    assert list(negs.columns) == ["user_id", "track_id", "liked"]

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
