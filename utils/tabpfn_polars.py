"""
Polars extension functions using TabPFN (via AutoGluon) for missing value
imputation and anomaly detection on tabular data.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import polars as pl
from autogluon.tabular import TabularPredictor


def _fit_tabpfn_predictor(
    train_df: pl.DataFrame,
    label: str,
    problem_type: str,
) -> TabularPredictor:
    """Train an AutoGluon TabularPredictor restricted to the TabPFN model."""
    pandas_df = train_df.to_pandas()
    predictor = TabularPredictor(
        label=label,
        problem_type=problem_type,
        verbosity=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictor.fit(
            pandas_df,
            hyperparameters={"TABPFN": {}},
            num_gpus=0,
        )
    return predictor


# ── 1. compute_missing_values ────────────────────────────────────────────────


def compute_missing_values(
    df: pl.DataFrame,
    col: str,
    context_columns: Optional[Sequence[str]] = None,
) -> pl.Series:
    """Impute missing (null) values in *col* using TabPFN.

    Parameters
    ----------
    df:
        Source data frame.
    col:
        Column whose nulls should be filled.
    context_columns:
        Optional list of feature columns to condition on.  When ``None``
        every other column in *df* is used.

    Returns
    -------
    pl.Series
        A copy of *col* with nulls replaced by TabPFN predictions.
    """
    if context_columns is None:
        context_columns = [c for c in df.columns if c != col]
    if not context_columns:
        raise ValueError("Need at least one context column for imputation.")

    used_cols = list(context_columns) + [col]
    working = df.select(used_cols)

    mask_missing = working[col].is_null()

    if not mask_missing.any():
        return df[col].clone()

    train = working.filter(~mask_missing)
    predict = working.filter(mask_missing).drop(col)

    if train.height == 0:
        raise ValueError(f"All values in '{col}' are null – nothing to learn from.")

    # Determine problem type from the column dtype
    if working[col].dtype in (pl.Utf8, pl.Categorical):
        problem_type = "multiclass"
    else:
        problem_type = "regression"

    predictor = _fit_tabpfn_predictor(train, label=col, problem_type=problem_type)
    preds = predictor.predict(predict.to_pandas())

    # Build the result series: original values where present, predictions where null
    result = df[col].to_list()
    pred_iter = iter(preds)
    for i in range(len(result)):
        if result[i] is None:
            result[i] = next(pred_iter)

    return pl.Series(col, result, dtype=df[col].dtype)


# ── 2. find_anomalies ───────────────────────────────────────────────────────


def find_anomalies(
    df: pl.DataFrame,
    col: str,
    context_columns: Optional[Sequence[str]] = None,
    threshold: float = 0.01,
) -> pl.DataFrame:
    """Flag anomalies in *col* by estimating how likely each value is under TabPFN.

    For categorical columns the predicted class‐probability for the true label
    is used directly.  For numeric columns the value is discretised into
    quantile bins so that TabPFN can output a probability per bin; the
    probability assigned to the bin that contains the true value is used.

    Parameters
    ----------
    df:
        Source data frame.
    col:
        Column to check for anomalies.
    context_columns:
        Optional feature columns.  Defaults to every column except *col*.
    threshold:
        Probability below which a value is considered anomalous (default 0.01).

    Returns
    -------
    pl.DataFrame
        Two‑column frame: the original *col* and a ``prob`` column with the
        estimated probability of each entry.  Rows whose ``prob < threshold``
        are the detected anomalies.
    """
    if context_columns is None:
        context_columns = [c for c in df.columns if c != col]
    if not context_columns:
        raise ValueError("Need at least one context column for anomaly detection.")

    used_cols = list(context_columns) + [col]
    working = df.select(used_cols)

    is_categorical = working[col].dtype in (pl.Utf8, pl.Categorical)

    if is_categorical:
        # TabPFN can give class probabilities directly
        predictor = _fit_tabpfn_predictor(
            working, label=col, problem_type="multiclass"
        )
        proba = predictor.predict_proba(working.drop(col).to_pandas())
        true_labels = working[col].to_list()
        probs = np.array(
            [proba.loc[i, label] if label in proba.columns else 0.0
             for i, label in enumerate(true_labels)]
        )
    else:
        # Discretise into quantile bins for probability estimation
        n_bins = min(10, working[col].n_unique())
        n_bins = max(n_bins, 2)

        values = working[col].to_numpy().astype(float)
        quantiles = np.nanquantile(values, np.linspace(0, 1, n_bins + 1))
        quantiles = np.unique(quantiles)
        if len(quantiles) < 3:
            # Nearly constant column – nothing is anomalous
            probs = np.ones(len(values))
            return pl.DataFrame({col: df[col], "prob": probs})

        bin_indices = np.digitize(values, quantiles[1:-1])  # 0 .. n_bins-1
        bin_labels = [str(b) for b in bin_indices]

        working_binned = working.drop(col).with_columns(
            pl.Series(col, bin_labels, dtype=pl.Utf8)
        )
        # reorder so label is last (same column set)
        working_binned = working_binned.select(
            [c for c in working_binned.columns if c != col] + [col]
        )

        predictor = _fit_tabpfn_predictor(
            working_binned, label=col, problem_type="multiclass"
        )
        proba = predictor.predict_proba(working_binned.drop(col).to_pandas())
        probs = np.array(
            [proba.loc[i, str(bin_indices[i])]
             if str(bin_indices[i]) in proba.columns else 0.0
             for i in range(len(bin_labels))]
        )

    return pl.DataFrame({col: df[col], "prob": pl.Series("prob", probs)})
