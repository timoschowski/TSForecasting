"""
Polars extension functions using TabPFN for missing value imputation and
anomaly detection on tabular data.

When TabPFN model weights are not available (e.g. no network access), the
functions fall back to scikit-learn estimators (RandomForest).
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import polars as pl


_tabpfn_available: bool | None = None


def _check_tabpfn() -> bool:
    """Return True if TabPFN can be loaded (model weights accessible)."""
    global _tabpfn_available
    if _tabpfn_available is not None:
        return _tabpfn_available
    try:
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier()
        # Trigger actual weight download with a tiny dummy fit
        clf.fit(np.array([[0, 1], [1, 0]]), np.array([0, 1]))
        _tabpfn_available = True
    except Exception:
        _tabpfn_available = False
    return _tabpfn_available


def _get_regressor():
    """Return a TabPFN regressor if available, else a scikit-learn fallback."""
    if _check_tabpfn():
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor()
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=100, random_state=0)


def _get_classifier():
    """Return a TabPFN classifier if available, else a scikit-learn fallback."""
    if _check_tabpfn():
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier()
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=100, random_state=0)


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
    predict_rows = working.filter(mask_missing).select(context_columns)

    if train.height == 0:
        raise ValueError(f"All values in '{col}' are null – nothing to learn from.")

    X_train = train.select(context_columns).to_numpy()
    y_train = train[col].to_numpy()
    X_pred = predict_rows.to_numpy()

    is_categorical = working[col].dtype in (pl.Utf8, pl.Categorical)

    model = _get_classifier() if is_categorical else _get_regressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_pred)

    # Build the result series
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

    For categorical columns the predicted class-probability for the true label
    is used directly.  For numeric columns the value is discretised into
    quantile bins so that TabPFN can output a probability per bin; the
    probability assigned to the bin containing the true value is reported.

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
        Two-column frame: the original *col* and a ``prob`` column with the
        estimated probability of each entry.  Rows whose ``prob < threshold``
        are the detected anomalies.
    """
    if context_columns is None:
        context_columns = [c for c in df.columns if c != col]
    if not context_columns:
        raise ValueError("Need at least one context column for anomaly detection.")

    used_cols = list(context_columns) + [col]
    working = df.select(used_cols)

    from sklearn.model_selection import cross_val_predict

    X = working.select(context_columns).to_numpy()
    is_categorical = working[col].dtype in (pl.Utf8, pl.Categorical)

    n_cv = min(5, len(X))

    if is_categorical:
        y = working[col].to_numpy()
        clf = _get_classifier()
        # Use cross-validated probabilities to avoid train-set memorisation
        proba = cross_val_predict(clf, X, y, cv=n_cv, method="predict_proba")
        clf_fit = _get_classifier()
        clf_fit.fit(X, y)
        classes = list(clf_fit.classes_)
        true_labels = working[col].to_list()
        probs = np.array([
            proba[i, classes.index(label)] if label in classes else 0.0
            for i, label in enumerate(true_labels)
        ])
    else:
        # Discretise into quantile bins for probability estimation
        values = working[col].to_numpy().astype(float)
        n_bins = min(10, len(np.unique(values[~np.isnan(values)])))
        n_bins = max(n_bins, 2)

        quantiles = np.nanquantile(values, np.linspace(0, 1, n_bins + 1))
        quantiles = np.unique(quantiles)
        if len(quantiles) < 3:
            probs = np.ones(len(values))
            return pl.DataFrame({col: df[col], "prob": probs})

        bin_indices = np.digitize(values, quantiles[1:-1])
        bin_labels = bin_indices.astype(str)

        clf = _get_classifier()
        proba = cross_val_predict(clf, X, bin_labels, cv=n_cv, method="predict_proba")
        clf_fit = _get_classifier()
        clf_fit.fit(X, bin_labels)
        classes = list(clf_fit.classes_)
        probs = np.array([
            proba[i, classes.index(str(bin_indices[i]))]
            if str(bin_indices[i]) in classes else 0.0
            for i in range(len(bin_labels))
        ])

    return pl.DataFrame({col: df[col], "prob": pl.Series("prob", probs)})
