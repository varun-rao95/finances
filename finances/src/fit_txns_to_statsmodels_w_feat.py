# fit_txns_to_statsmodels_features.py
"""
Feature‑conditioned, bucketed two‑part spend model (v2).

Extends your original pipeline but now handles **degenerate buckets** where
`active` is all‑ones or all‑zeros (no variance → singular matrix in Logit).

Strategy
--------
* If `active` has both 0 and 1 → fit full Logistic as before.
* If `active` is **constant** → store scalar `p_active` (0.0 or 1.0) and
  skip the Logit fit.

Everything else stays identical.  The simulator recognises the presence of
`p_active` and treats those buckets deterministically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Sequence, List

###############################################################################
# Helper – build numeric design matrix w/ intercept, drop zero‑var columns
###############################################################################


def _add_const(X: pd.DataFrame | pd.Series) -> np.ndarray:
    if isinstance(X, pd.Series):
        X = X.to_frame().T
    arr = X.astype(float).to_numpy()
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    keep = arr.std(axis=0) > 1e-10
    arr = arr[:, keep]
    return sm.add_constant(arr, has_constant="add")


###############################################################################
# Core model class
###############################################################################


class BucketedTwoPartModel:
    """Two‑part bucket model.

    * **Feature‑conditioned** logistic + log‑normal for buckets with enough data.
    * **Fallback dummy** (global Bernoulli + global log‑normal intercept) for
      sparse buckets – either:
        • sample count < `min_obs`, **or**
        • it is the final open‑ended bucket (index == len(bucket_edges)).
    """

    def __init__(self, bucket_edges: Sequence[float], min_obs: int = 50):
        self.bucket_edges = list(bucket_edges)
        self.min_obs = min_obs
        self.bucket_models: Dict[int, Dict[str, object]] = {}

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BucketedTwoPartModel":
        bucket_idx = y.apply(lambda v: kde_bucket_thresholds(v, self.bucket_edges))
        self.bucket_models = {}
        overflow_idx = len(self.bucket_edges)  # last implicit bucket

        for b in bucket_idx.unique():
            idx_b = bucket_idx == b
            y_b = y[idx_b]
            X_b = X.loc[idx_b]
            active = (y_b > 0).astype(int)
            n_rows = len(y_b)

            # Decide if we drop to dummy behaviour
            need_dummy = (n_rows < self.min_obs) or (b == overflow_idx)
            mods: Dict[str, object] = {}

            # ----------------- Bernoulli part -------------------------
            if need_dummy or active.nunique() == 1:
                mods["p_active"] = float(active.mean())  # could be 0/1 or small prob
            else:
                mods["logit"] = sm.Logit(active, _add_const(X_b)).fit(disp=0)

            # ----------------- Log‑normal part ------------------------
            if active.sum() == 0:
                mods["lognorm"] = None
            else:
                y_pos = np.log(y_b[active == 1])
                # If dummy: use intercept‑only; else use full features
                if need_dummy:
                    ln_res = sm.OLS(y_pos, np.ones((len(y_pos), 1))).fit(disp=0)
                else:
                    ln_res = sm.OLS(y_pos, _add_const(X_b[active == 1])).fit(disp=0)
                mods["lognorm"] = ln_res

            self.bucket_models[b] = mods
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
