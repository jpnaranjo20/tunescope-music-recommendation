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

import numpy as np
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


def transform(encoder: Encoder, pairs: pd.DataFrame) -> pd.DataFrame:
    """Given (user_id, track_id) pairs, return a feature matrix.

    Output rows are aligned with ``pairs``. Cold-pair interaction values
    (user has never played the track / artist) come back as 0 for counts
    and ``NaN`` for ``ut_days_since_last_play``. LightGBM handles
    missing values natively.
    """
    out = pairs[["user_id", "track_id"]].copy().reset_index(drop=True)

    out = out.merge(encoder.user_features, left_on="user_id", right_index=True, how="left")
    out = out.merge(encoder.track_features, left_on="track_id", right_index=True, how="left")

    ut_keys = list(zip(out.user_id.tolist(), out.track_id.tolist(), strict=True))
    out["ut_prior_play_count"] = [
        encoder.user_track_plays.get((int(u), int(t)), 0) for u, t in ut_keys
    ]

    cutoffs = [encoder.split_cutoffs.get(int(u)) for u, _ in ut_keys]
    last_ts = [encoder.user_track_last_ts.get((int(u), int(t))) for u, t in ut_keys]
    days: list[float] = []
    for ts, cutoff in zip(last_ts, cutoffs, strict=True):
        if ts is None or cutoff is None:
            days.append(float("nan"))
        else:
            days.append((cutoff - ts).total_seconds() / 86400.0)
    out["ut_days_since_last_play"] = days

    ua_keys = list(zip(out.user_id.tolist(), out.t_artist_id.tolist(), strict=True))
    out["ua_n_plays_of_artist"] = [
        encoder.user_artist_plays.get((int(u), int(a)), 0) for u, a in ua_keys
    ]

    return out


def sample_negatives(
    encoder: Encoder,
    positive_pairs: pd.DataFrame,
    *,
    k: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """For each positive (u, t), sample ``k`` random tracks the user has
    never played in train. Returns a DataFrame with columns
    ``(user_id, track_id, liked=0)``.

    Per-user negatives are sampled WITHOUT replacement to avoid trivial
    duplicates. If a user has played more than half the catalog,
    rejection sampling stalls — fall back to ``np.setdiff1d`` then
    ``rng.choice``.
    """
    rng = np.random.default_rng(seed)

    user_played: dict[int, set[int]] = {}
    for u, t in encoder.user_track_plays:
        user_played.setdefault(int(u), set()).add(int(t))

    out_users: list[int] = []
    out_tracks: list[int] = []

    for u in positive_pairs.user_id.tolist():
        u = int(u)
        played = user_played.get(u, set())
        if encoder.n_tracks - len(played) < k:
            unplayed_count = encoder.n_tracks - len(played)
            if unplayed_count <= 0:
                continue
            unplayed = np.setdiff1d(np.arange(encoder.n_tracks), list(played))
            chosen = rng.choice(unplayed, size=unplayed_count, replace=False)
        elif len(played) > encoder.n_tracks * 0.5:
            unplayed = np.setdiff1d(np.arange(encoder.n_tracks), list(played))
            chosen = rng.choice(unplayed, size=k, replace=False)
        else:
            chosen_set: set[int] = set()
            while len(chosen_set) < k:
                cands = rng.integers(0, encoder.n_tracks, size=k * 2)
                for c in cands:
                    c = int(c)
                    if c not in played and c not in chosen_set:
                        chosen_set.add(c)
                        if len(chosen_set) == k:
                            break
            chosen = np.array(sorted(chosen_set))

        for c in chosen:
            out_users.append(u)
            out_tracks.append(int(c))

    return pd.DataFrame(
        {
            "user_id": out_users,
            "track_id": out_tracks,
            "liked": [0] * len(out_users),
        }
    )
