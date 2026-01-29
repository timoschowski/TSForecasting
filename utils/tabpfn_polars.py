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


# ── 3. find_similar_rows ─────────────────────────────────────────────────────


def _get_embeddings(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return row embeddings using TabPFN if available, else sklearn leaf nodes."""
    if _check_tabpfn():
        from tabpfn import TabPFNClassifier

        clf = TabPFNClassifier()
        clf.fit(X, y)
        return clf.get_embeddings(X, data_source="train")

    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)
    # Use leaf-node indices across all trees as an embedding
    leaf_indices = clf.apply(X)  # (n_samples, n_trees)
    return leaf_indices.astype(float)


def find_similar_rows(
    df: pl.DataFrame,
    row_index: int,
    k: int = 5,
    context_columns: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """Find the *k* most similar rows to a given row using TabPFN embeddings.

    A dummy classification target is constructed from quantile bins of the
    first context column so that TabPFN can produce embeddings.  Similarity
    is measured by cosine distance in the embedding space.

    Parameters
    ----------
    df:
        Source data frame.
    row_index:
        Index of the query row.
    k:
        Number of nearest neighbours to return (default 5).
    context_columns:
        Columns to use for computing embeddings.  Defaults to all columns.

    Returns
    -------
    pl.DataFrame
        The *k* closest rows (excluding the query row itself), with an extra
        ``similarity`` column (1 = identical, 0 = orthogonal).
    """
    if context_columns is None:
        context_columns = list(df.columns)
    if not context_columns:
        raise ValueError("Need at least one column for embeddings.")

    X = df.select(context_columns).to_numpy().astype(float)

    # Build a synthetic classification target for the embedding model
    ref_col = X[:, 0]
    n_bins = min(5, len(np.unique(ref_col)))
    n_bins = max(n_bins, 2)
    quantiles = np.unique(np.nanquantile(ref_col, np.linspace(0, 1, n_bins + 1)))
    if len(quantiles) < 3:
        y_dummy = np.zeros(len(X), dtype=int)
    else:
        y_dummy = np.digitize(ref_col, quantiles[1:-1])

    embeddings = _get_embeddings(X, y_dummy)

    # Cosine similarity
    query = embeddings[row_index]
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query)
    norms = np.where(norms == 0, 1e-10, norms)
    similarities = embeddings @ query / norms

    # Exclude the query row itself, pick top-k
    similarities[row_index] = -np.inf
    top_k_indices = np.argsort(similarities)[::-1][:k]

    result = df[top_k_indices.tolist()]
    return result.with_columns(
        pl.Series("similarity", similarities[top_k_indices])
    )


# ── 4. extrapolate_dataframe ─────────────────────────────────────────────────


def extrapolate_dataframe(
    df: pl.DataFrame,
    n_rows: int,
    context_columns: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """Generate *n_rows* new synthetic rows by extrapolating from existing data.

    Each column is predicted in turn, conditioned on the other columns.
    New feature vectors are created by sampling existing rows and adding
    Gaussian noise, then each column is re-predicted by TabPFN (or fallback)
    to produce coherent synthetic records.

    Parameters
    ----------
    df:
        Source data frame (numeric columns only).
    n_rows:
        Number of new rows to generate.
    context_columns:
        Columns to include.  Defaults to all columns.

    Returns
    -------
    pl.DataFrame
        A data frame with *n_rows* synthetic rows and the same schema as the
        selected columns.
    """
    if context_columns is None:
        context_columns = list(df.columns)
    if len(context_columns) < 2:
        raise ValueError("Need at least two columns to extrapolate.")

    data = df.select(context_columns).to_numpy().astype(float)
    n_orig, n_cols = data.shape

    rng = np.random.default_rng(seed=0)

    # Bootstrap sample + small perturbation as seed rows
    sample_indices = rng.choice(n_orig, size=n_rows, replace=True)
    col_stds = np.nanstd(data, axis=0)
    col_stds = np.where(col_stds == 0, 1.0, col_stds)
    noise = rng.normal(scale=col_stds * 0.1, size=(n_rows, n_cols))
    synthetic = data[sample_indices] + noise

    # Refine each column by predicting it from the others
    for col_idx in range(n_cols):
        feature_idx = [j for j in range(n_cols) if j != col_idx]
        X_train = data[:, feature_idx]
        y_train = data[:, col_idx]
        X_pred = synthetic[:, feature_idx]

        model = _get_regressor()
        model.fit(X_train, y_train)
        synthetic[:, col_idx] = model.predict(X_pred)

    return pl.DataFrame(
        {col: synthetic[:, i] for i, col in enumerate(context_columns)},
        schema={col: df[col].dtype for col in context_columns},
    )
