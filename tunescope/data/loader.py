"""Data loader: parquet on disk → typed in-memory DataFrames.

Source-agnostic by design — works with both ``data/raw/synthetic/`` and
the future ``data/raw/lastfm-1k/`` because both have the same schema
(see ``DATA.md``). Applies one transformation: drops users with fewer
than ``min_plays`` plays (and the corresponding play rows) so cold-start
users don't pollute training. ``tracks`` is returned unchanged so the
popularity prior can still see cold tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Dataset:
    users: pd.DataFrame
    tracks: pd.DataFrame
    plays: pd.DataFrame


def load(source_dir: Path | str, *, min_plays: int = 5) -> Dataset:
    """Read users / tracks / plays parquet from ``source_dir`` and apply
    the min-plays filter.

    Returns a Dataset whose internals are independent of which source
    directory the data came from.
    """
    source_dir = Path(source_dir)
    users = pd.read_parquet(source_dir / "users.parquet")
    tracks = pd.read_parquet(source_dir / "tracks.parquet")
    plays = pd.read_parquet(source_dir / "plays.parquet")

    counts = plays.user_id.value_counts()
    eligible = counts[counts >= min_plays].index
    users = users[users.user_id.isin(eligible)].reset_index(drop=True)
    plays = plays[plays.user_id.isin(eligible)].reset_index(drop=True)

    return Dataset(users=users, tracks=tracks, plays=plays)
