"""Feature encoder for the music recommender.

Provides:
    split_by_time      — per-user time-based train/test split
    build_labels       — implicit-feedback "liked" labels
    fit_encoder        — build per-user / per-track / per-pair lookups
    transform          — apply encoder to (user, track) pairs
    sample_negatives   — sample unplayed tracks per positive

The encoder fits ONLY on training data (test plays are never passed in).
All dict structures are built with sorted-key insertion so joblib pickle
hashes are deterministic across runs.
"""

from __future__ import annotations

import pandas as pd


def split_by_time(
    plays: pd.DataFrame, *, holdout_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user time-based split.

    For each user, sort their plays by ts ascending and take the first
    ``floor((1 - holdout_frac) * n_user_plays)`` rows as train, the rest
    as test. Both outputs are reset_index'd for downstream cleanliness.
    """
    sorted_plays = plays.sort_values(["user_id", "ts"], kind="stable").reset_index(drop=True)
    user_counts = sorted_plays.groupby("user_id").size()
    train_sizes = (user_counts * (1 - holdout_frac)).astype(int)

    rank = sorted_plays.groupby("user_id").cumcount()
    user_train_size = sorted_plays.user_id.map(train_sizes)
    is_train = rank < user_train_size

    train = sorted_plays[is_train].reset_index(drop=True)
    test = sorted_plays[~is_train].reset_index(drop=True)
    return train, test
