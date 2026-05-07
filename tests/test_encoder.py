"""Tests for the feature encoder, label builder, time-based split, and
negative sampler. Fixtures use a small deterministic synthetic dataset
so tests stay fast (~1s each).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.generate_synthetic import GenConfig, generate
from tunescope.data.loader import load
from tunescope.features.encode import split_by_time

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
