# fit_txns_to_statsmodels_features.py
"""
Feature‑conditioned, bucketed two‑part spend model.

This module *extends* your original `fit_txns_to_statsmodels.py` pipeline
without touching its univariate logic.  Import what you need from the
original file (e.g. `kde_bucket_thresholds`, `fit_two_part_gamma`,
`fit_two_part_lognorm`) and add **feature‑aware** Bernoulli + Lognormal
models per bucket.

Quick usage -------------------------------------------------------------------
>>> from fit_txns_to_statsmodels_features import (
        BucketedTwoPartModel,  evaluate_splits
    )
>>> raw = load_data()                                 # from original module
>>> frame = make_feature_target_frame(raw)            # from txn_features.py
>>> splits = generate_weekly_splits(frame)            # txn_features.py
>>> bucket_edges = [...]                              # your thresholds
>>> results = evaluate_splits(frame, bucket_edges, splits, n_mc=500)

`results` is a DataFrame with MAE / RMSE per split; inspect and aggregate
as you like.
"""
from __future__ import annotations

from fit_txns_to_statsmodels import kde_bucket_thresholds
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Sequence, List

# ---------------------------------------------------------------------------
# Helper — simple design‑matrix builder (add constant)
# ---------------------------------------------------------------------------


def _add_const(X: pd.DataFrame | pd.Series) -> np.ndarray:
    """
    Ensure numeric 2-D array and prepend constant column for statsmodels.
    """
    if isinstance(X, pd.Series):
        X = X.to_frame().T  # make it 2-D for a single row case
    arr = X.astype(float).to_numpy()  # coerce bool/int/uint8 → float64
    return sm.add_constant(arr, has_constant="add")


# ---------------------------------------------------------------------------
# Core model class
# ---------------------------------------------------------------------------


class BucketedTwoPartModel:
    """Bernoulli(logit)  +  LogNormal(OLS)  per bucket, conditioned on features."""

    def __init__(self, bucket_edges: Sequence[float]):
        self.bucket_edges = list(bucket_edges)
        self.bucket_models: Dict[int, Dict[str, object]] = {}

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BucketedTwoPartModel":
        """Fit per‑bucket two‑part models using feature matrix X and total‑spend y."""
        # Map each spend value -> bucket index (like original code)
        bucket_idx = y.apply(lambda v: kde_bucket_thresholds(v, self.bucket_edges))
        self.bucket_models = {}
        for b in bucket_idx.unique():
            idx_b = bucket_idx == b
            y_b = y[idx_b]
            X_b = X.loc[idx_b]

            # Binary activity indicator (any spend > 0 in that bucket)
            active = (y_b > 0).astype(int)
            try:
                logit_res = sm.Logit(active, _add_const(X_b)).fit(disp=0)
            except:
                print("Bucket", b, "shape", X_b.shape)
                print("Cols variance:\n", X_b.var())
                print("active value counts:\n", active.value_counts())

                const_in_active = X_b[active == 1].var() < 1e-10
                print(
                    "Zero-var cols inside active==1 subset:\n",
                    const_in_active[const_in_active].index,
                )

                raise

            # If we never see positive spend for a bucket, skip sigma/mu
            if active.sum() == 0:
                self.bucket_models[b] = {"logit": logit_res, "lognorm": None}
                continue

            # Log(amount) for positive rows
            y_pos = np.log(y_b[active == 1])
            X_pos = X_b[active == 1]
            ln_res = sm.OLS(y_pos, _add_const(X_pos)).fit(disp=0)

            self.bucket_models[b] = {"logit": logit_res, "lognorm": ln_res}
        return self

    # -------------------------------------------------------------------
    def simulate_day(self, x_row: pd.Series, n_mc: int = 1000) -> np.ndarray:
        """Return *n_mc* Monte Carlo draws of total spend for one day."""
        draws = np.zeros(n_mc)
        for b, mods in self.bucket_models.items():
            logit_p = mods["logit"].predict(_add_const(x_row.to_frame().T))[0]
            active = np.random.rand(n_mc) < logit_p
            if mods["lognorm"] is None:
                continue  # never positive in train
            mu = mods["lognorm"].predict(_add_const(x_row.to_frame().T))[0]
            sigma = np.sqrt(mods["lognorm"].scale)
            bucket_spend = np.where(
                active, np.random.lognormal(mean=mu, sigma=sigma, size=n_mc), 0.0
            )
            draws += bucket_spend
        return draws

    # -------------------------------------------------------------------
    def predict_expected(self, X: pd.DataFrame, n_mc: int = 1000) -> np.ndarray:
        """Return expected spend (mean of simulations) for each row in X."""
        out = np.empty(len(X))
        for i, (_, row) in enumerate(X.iterrows()):
            out[i] = self.simulate_day(row, n_mc=n_mc).mean()
        return out


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def _mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def _rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def evaluate_splits(
    frame: pd.DataFrame,
    bucket_edges: Sequence[float],
    splits: List[Dict[str, pd.Index]],
    feature_cols: Sequence[str] | None = None,
    n_mc: int = 500,
) -> pd.DataFrame:
    """Train + test the BucketedTwoPartModel for each provided split.

    *frame* should already have the column `target_next_day` and the desired
    feature columns.  If `feature_cols` is None, we use all columns except the
    target.
    """
    feature_cols = feature_cols or [c for c in frame.columns if c != "target_next_day"]
    records = []
    for i, split in enumerate(splits):
        X_train = frame.loc[split["train_idx"], feature_cols]
        y_train = frame.loc[split["train_idx"], "target_next_day"]
        X_test = frame.loc[split["test_idx"], feature_cols]
        y_test = frame.loc[split["test_idx"], "target_next_day"]

        model = BucketedTwoPartModel(bucket_edges).fit(X_train, y_train)
        y_pred = model.predict_expected(X_test, n_mc=n_mc)

        records.append(
            {
                "split": i,
                "anchor": split.get("anchor", "chron"),
                "mae": _mae(y_test.values, y_pred),
                "rmse": _rmse(y_test.values, y_pred),
                "n_test": len(y_test),
            }
        )
    return pd.DataFrame.from_records(records)
