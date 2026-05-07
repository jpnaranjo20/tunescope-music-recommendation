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

from dataclasses import dataclass

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


def build_labels(plays: pd.DataFrame, *, threshold: int = 3) -> pd.DataFrame:
    """Return (user_id, track_id, liked=1) for every (user, track) pair
    appearing at least ``threshold`` times in ``plays``."""
    counts = plays.groupby(["user_id", "track_id"]).size().rename("n").reset_index()
    liked = counts[counts.n >= threshold].drop(columns="n").copy()
    liked["liked"] = 1
    return liked.reset_index(drop=True)


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
    plays_w_artist = train_plays.merge(tracks[["track_id", "artist_id"]], on="track_id", how="left")

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
        int(uid): ts for uid, ts in sorted(train_plays.groupby("user_id")["ts"].max().items())
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
