# txn_features.py
"""
Feature engineering & time‑aware train/test splitting utilities
for next‑day (and longer‑horizon) spend prediction.

Designed to plug into the existing two‑part bucket simulator: you build a
daily feature frame → obtain train/test slices → hand the train slice to
bucket‑level Bernoulli/Lognormal trainers.

Assumptions
-----------
* Raw transactions DataFrame has at minimum columns:
    - Date (datetime or string parsable)
    - Amount (float)
* Each row is a single transaction.
* Caller decides whether to pre‑filter categories, outliers, etc.

Typical usage
-------------
>>> from txn_features import build_daily_feature_frame, generate_weekly_splits
>>> df_raw = load_data(...)
>>> feats = build_daily_feature_frame(df_raw)
>>> target = add_next_day_target(feats)  # adds column "target_next_day"
>>> splits = generate_weekly_splits(target, anchor_days=["Monday","Wednesday","Saturday"])

Each element in `splits` is a dict with keys `train_idx`, `test_idx`,
`cv_idx` that you can use to slice the feature frame for model fitting &
evaluation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Iterable

###############################################################################
# 1.  Daily feature engineering
###############################################################################


def _aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw transaction rows into **one row per day** with primitive
    aggregations we can build richer features from.
    Returns a DataFrame indexed by daily date with columns:
        total_spend, num_txns, max_txn, avg_txn
    """
    if "Date" not in df.columns or "Amount" not in df.columns:
        raise ValueError("Input DataFrame must contain Date and Amount columns.")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    daily = (
        df.groupby("Date")["Amount"]
        .agg(total_spend="sum", num_txns="size", max_txn="max", avg_txn="mean")
        .sort_index()
    )

    # ensure no missing days – fill with zeros
    full_range = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0.0)
    daily.index.name = "Date"
    return daily


def _rolling(series: pd.Series, window: int, func: str = "mean") -> pd.Series:
    """Convenience wrapper with min_periods = 1 so early days get values."""
    return getattr(series.rolling(window=window, min_periods=1), func)()


def build_daily_feature_frame(df_txns: pd.DataFrame) -> pd.DataFrame:
    """Return a feature DataFrame indexed by date.

    Core features implemented (can be expanded later):
    -----------------------------------------------
    * total_spend, num_txns, max_txn, avg_txn
    * rolling_3d_avg_spend, rolling_7d_avg_spend, rolling_30d_avg_spend
    * days_since_high_spend_day (>90th percentile)
    * consecutive_days_below_threshold (total_spend < 50th percentile)
    * weekday (0‑Mon … 6‑Sun) as categorical one‑hot columns
    * day_of_month
    """
    daily = _aggregate_daily(df_txns)

    # Rolling means
    daily["rolling_3d_avg_spend"] = _rolling(daily["total_spend"], 3)
    daily["rolling_7d_avg_spend"] = _rolling(daily["total_spend"], 7)
    daily["rolling_30d_avg_spend"] = _rolling(daily["total_spend"], 30)

    # High‑spend threshold (static percentile over entire history)
    high_threshold = daily["total_spend"].quantile(0.90)
    low_threshold = daily["total_spend"].quantile(0.50)

    # Days since last high‑spend
    last_high = (~(daily["total_spend"] > high_threshold)).astype(int).cumsum()
    daily["days_since_high_spend"] = last_high - last_high.where(
        daily["total_spend"] > high_threshold
    ).ffill().fillna(0).astype(int)

    # Consecutive days below low_threshold
    below = daily["total_spend"] < low_threshold
    daily["consecutive_days_below_threshold"] = (
        below.groupby((~below).cumsum()).cumcount() + 1
    )
    daily.loc[~below, "consecutive_days_below_threshold"] = 0

    # Calendar features
    daily["weekday"] = daily.index.weekday  # 0‑Mon … 6‑Sun
    daily["day_of_month"] = daily.index.day
    # One‑hot weekday
    weekday_dummies = pd.get_dummies(daily["weekday"], prefix="wd", drop_first=True)
    daily = pd.concat([daily, weekday_dummies], axis=1)

    return daily


###############################################################################
# 2.  Targets
###############################################################################


def add_next_day_target(
    df_daily: pd.DataFrame, target_col: str = "target_next_day"
) -> pd.DataFrame:
    """Adds next‑day spend target column, dropping last row with NaN target."""
    df = df_daily.copy()
    df[target_col] = df["total_spend"].shift(-1)
    return df.dropna(subset=[target_col])


###############################################################################
# 3.  Time‑based splits
###############################################################################


def chronological_75205_split(df: pd.DataFrame) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """Simple 75/20/5 chronological split.

    Returns train_idx, test_idx, cv_idx as Index objects ready for .loc slicing."""
    n = len(df)
    train_end = int(0.75 * n)
    test_end = int(0.95 * n)
    train_idx = df.index[:train_end]
    test_idx = df.index[train_end:test_end]
    cv_idx = df.index[test_end:]
    return train_idx, test_idx, cv_idx


def generate_weekly_splits(
    df: pd.DataFrame,
    anchor_days: Iterable[str] = ("Monday", "Wednesday", "Saturday"),
    train_days: int = 5,
    test_days: int = 1,
    cv_days: int = 2,
) -> List[Dict[str, pd.Index]]:
    """Return list of rolling split definitions based on weekly blocks.

    Each split is a dict with keys train_idx, test_idx, cv_idx.
    """
    splits: List[Dict[str, pd.Index]] = []
    df = df.sort_index()
    for anchor in anchor_days:
        # find all rows where weekday matches anchor
        anchor_dates = df[df.index.day_name() == anchor].index
        for start in anchor_dates:
            train_start = start
            train_end = train_start + pd.Timedelta(days=train_days - 1)
            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.Timedelta(days=test_days - 1)
            cv_start = test_end + pd.Timedelta(days=1)
            cv_end = cv_start + pd.Timedelta(days=cv_days - 1)

            if cv_end > df.index[-1]:
                break  # stop when we run out of data
            splits.append(
                {
                    "anchor": anchor,
                    "train_idx": df.index[
                        (df.index >= train_start) & (df.index <= train_end)
                    ],
                    "test_idx": df.index[
                        (df.index >= test_start) & (df.index <= test_end)
                    ],
                    "cv_idx": df.index[(df.index >= cv_start) & (df.index <= cv_end)],
                }
            )
    return splits


###############################################################################
# 4.  Convenience wrappers
###############################################################################


def make_feature_target_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Pipe convenience: raw → features → target column → returns frame ready for splits."""
    return add_next_day_target(build_daily_feature_frame(raw_df))


###############################################################################
# __main__ quick demo (optional)
###############################################################################
if __name__ == "__main__":
    # quick smoke test with a tiny dummy DataFrame
    rng = pd.date_range("2025-01-01", periods=100, freq="D")
    dummy = pd.DataFrame(
        {
            "Date": np.repeat(rng, 3),  # 3 txns per day
            "Amount": np.random.lognormal(mean=3, sigma=0.5, size=len(rng) * 3),
        }
    )

    frame = make_feature_target_frame(dummy)
    print(frame.head())

    train_idx, test_idx, cv_idx = chronological_75205_split(frame)
    print(
        "Train size", len(train_idx), "Test size", len(test_idx), "CV size", len(cv_idx)
    )

    splits = generate_weekly_splits(frame)
    print("Generated", len(splits), "weekly splits")
