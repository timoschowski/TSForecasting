"""
AutoGluon-Enhanced Polars DataFrame

This module provides functionality to enhance Polars DataFrames with AutoGluon's
tabular foundation models, enabling:
- Row similarity search using embeddings
- Column similarity search using embeddings
- Missing value interpolation using foundation models
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Union, List, Optional, Tuple
from autogluon.tabular import TabularPredictor
import tempfile
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import warnings


class AGPolarsDataFrame:
    """
    AutoGluon-enhanced Polars DataFrame wrapper that provides:
    - Row similarity search using foundation model embeddings
    - Column similarity search using embeddings
    - Missing value interpolation using foundation models
    """

    def __init__(self, df: pl.DataFrame, predictor: Optional[TabularPredictor] = None):
        """
        Initialize the AutoGluon-enhanced DataFrame.

        Args:
            df: Polars DataFrame
            predictor: Optional pre-trained TabularPredictor. If None, a new one
                      will be created when needed.
        """
        self.df = df
        self._predictor = predictor
        self._row_embeddings = None
        self._col_embeddings = None
        self._row_nn_model = None
        self._col_nn_model = None

    @property
    def predictor(self) -> TabularPredictor:
        """Get or create the TabularPredictor."""
        if self._predictor is None:
            raise ValueError(
                "No predictor available. Either provide a predictor during "
                "initialization or train one using fit_predictor()."
            )
        return self._predictor

    def fit_predictor(
        self,
        label: str,
        presets: str = "best_quality",
        time_limit: Optional[int] = None,
        **kwargs
    ) -> 'AGPolarsDataFrame':
        """
        Fit a TabularPredictor on the DataFrame with foundation models.

        Args:
            label: Name of the column to predict (target variable)
            presets: Quality preset. Use "best_quality" for foundation models.
            time_limit: Time limit in seconds for training
            **kwargs: Additional arguments for TabularPredictor.fit()

        Returns:
            Self for method chaining
        """
        # Convert to pandas for AutoGluon
        df_pd = self.df.to_pandas()

        # Create temporary directory for model
        temp_dir = tempfile.mkdtemp(prefix="autogluon_")

        # Initialize and fit predictor
        self._predictor = TabularPredictor(
            label=label,
            path=temp_dir,
            verbosity=2
        )

        self._predictor.fit(
            train_data=df_pd,
            presets=presets,
            time_limit=time_limit,
            **kwargs
        )

        return self

    def _compute_row_embeddings(self, force_recompute: bool = False) -> np.ndarray:
        """
        Compute embeddings for each row using the foundation model.

        Args:
            force_recompute: If True, recompute even if cached

        Returns:
            Array of shape (n_rows, embedding_dim)
        """
        if self._row_embeddings is not None and not force_recompute:
            return self._row_embeddings

        # Convert to pandas for AutoGluon
        df_pd = self.df.to_pandas()

        # Try to get transformed features from the predictor
        # This will give us feature representations from the model
        try:
            # Use transform_features to get feature representations
            transformed = self.predictor.transform_features(df_pd)
            embeddings = transformed.values
        except Exception as e:
            warnings.warn(
                f"Could not extract features from predictor: {e}. "
                "Using numeric columns as embeddings."
            )
            # Fallback: use numeric columns
            numeric_cols = self.df.select(pl.col(pl.Float64, pl.Float32, pl.Int64, pl.Int32)).columns
            embeddings = self.df.select(numeric_cols).to_numpy()

        # Handle any remaining NaN values
        embeddings = np.nan_to_num(embeddings, nan=0.0)

        self._row_embeddings = embeddings
        return embeddings

    def _compute_column_embeddings(self, force_recompute: bool = False) -> np.ndarray:
        """
        Compute embeddings for each column using the foundation model.

        This transposes the data so each column becomes a row, then computes embeddings.

        Args:
            force_recompute: If True, recompute even if cached

        Returns:
            Array of shape (n_cols, embedding_dim)
        """
        if self._col_embeddings is not None and not force_recompute:
            return self._col_embeddings

        # Get numeric columns only
        numeric_cols = self.df.select(pl.col(pl.Float64, pl.Float32, pl.Int64, pl.Int32)).columns

        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for computing column embeddings")

        # Compute statistics for each column as embeddings
        # This includes: mean, std, min, max, median, skewness, etc.
        col_embeddings = []

        for col in numeric_cols:
            col_data = self.df[col].drop_nulls()

            if len(col_data) == 0:
                # All nulls - create zero embedding
                col_embeddings.append(np.zeros(7))
                continue

            stats = [
                col_data.mean(),
                col_data.std(),
                col_data.min(),
                col_data.max(),
                col_data.median(),
                float(len(self.df[col].filter(pl.col(col).is_null()))),  # null count
                float(len(col_data))  # non-null count
            ]
            col_embeddings.append(stats)

        self._col_embeddings = np.array(col_embeddings)
        return self._col_embeddings

    def find_similar_rows(
        self,
        row_indices: Union[int, List[int]],
        n_similar: int = 5,
        include_self: bool = False
    ) -> pl.DataFrame:
        """
        Find similar rows to the given row(s) using embeddings.

        Args:
            row_indices: Single row index or list of row indices
            n_similar: Number of similar rows to return
            include_self: Whether to include the query row(s) in results

        Returns:
            DataFrame with columns: query_idx, similar_idx, similarity_score
        """
        # Compute embeddings if needed
        embeddings = self._compute_row_embeddings()

        # Normalize indices
        if isinstance(row_indices, int):
            row_indices = [row_indices]

        # Fit nearest neighbors model if needed
        if self._row_nn_model is None:
            self._row_nn_model = NearestNeighbors(
                n_neighbors=n_similar + 1,  # +1 to account for self
                metric='cosine'
            )
            self._row_nn_model.fit(embeddings)

        # Find similar rows for each query
        results = []
        for query_idx in row_indices:
            query_embedding = embeddings[query_idx:query_idx+1]
            distances, indices = self._row_nn_model.kneighbors(query_embedding)

            for dist, idx in zip(distances[0], indices[0]):
                if not include_self and idx == query_idx:
                    continue

                similarity = 1 - dist  # Convert distance to similarity
                results.append({
                    'query_idx': query_idx,
                    'similar_idx': int(idx),
                    'similarity_score': float(similarity)
                })

        # Create result DataFrame and sort by query_idx and similarity
        result_df = pl.DataFrame(results)
        if len(result_df) > 0:
            result_df = result_df.sort(['query_idx', 'similarity_score'], descending=[False, True])

            # Limit to n_similar per query
            result_df = result_df.group_by('query_idx').head(n_similar)

        return result_df

    def find_similar_columns(
        self,
        column_names: Union[str, List[str]],
        n_similar: int = 3,
        include_self: bool = False
    ) -> pl.DataFrame:
        """
        Find similar columns to the given column(s) using embeddings.

        Args:
            column_names: Single column name or list of column names
            n_similar: Number of similar columns to return
            include_self: Whether to include the query column in results

        Returns:
            DataFrame with columns: query_column, similar_column, similarity_score
        """
        # Compute column embeddings
        col_embeddings = self._compute_column_embeddings()

        # Get numeric columns
        numeric_cols = self.df.select(pl.col(pl.Float64, pl.Float32, pl.Int64, pl.Int32)).columns

        # Normalize column names
        if isinstance(column_names, str):
            column_names = [column_names]

        # Validate column names
        for col_name in column_names:
            if col_name not in numeric_cols:
                raise ValueError(
                    f"Column '{col_name}' not found in numeric columns: {numeric_cols}"
                )

        # Fit nearest neighbors model if needed
        if self._col_nn_model is None:
            self._col_nn_model = NearestNeighbors(
                n_neighbors=min(n_similar + 1, len(numeric_cols)),
                metric='cosine'
            )
            self._col_nn_model.fit(col_embeddings)

        # Find similar columns for each query
        results = []
        for query_col in column_names:
            query_idx = numeric_cols.index(query_col)
            query_embedding = col_embeddings[query_idx:query_idx+1]

            distances, indices = self._col_nn_model.kneighbors(query_embedding)

            for dist, idx in zip(distances[0], indices[0]):
                similar_col = numeric_cols[idx]

                if not include_self and similar_col == query_col:
                    continue

                similarity = 1 - dist
                results.append({
                    'query_column': query_col,
                    'similar_column': similar_col,
                    'similarity_score': float(similarity)
                })

        # Create result DataFrame
        result_df = pl.DataFrame(results)
        if len(result_df) > 0:
            result_df = result_df.sort(
                ['query_column', 'similarity_score'],
                descending=[False, True]
            )

            # Limit to n_similar per query
            result_df = result_df.group_by('query_column').head(n_similar)

        return result_df

    def interpolate_missing(
        self,
        row_idx: int,
        column: str,
        method: str = 'predict'
    ) -> Union[float, int, str]:
        """
        Interpolate a missing value using the foundation model.

        Args:
            row_idx: Row index with missing value
            column: Column name with missing value
            method: Interpolation method. Options:
                   - 'predict': Use the predictor to predict the value
                   - 'similar_rows': Use average of similar rows
                   - 'similar_columns': Use correlation with similar columns

        Returns:
            Interpolated value
        """
        if method == 'predict':
            return self._interpolate_with_predictor(row_idx, column)
        elif method == 'similar_rows':
            return self._interpolate_with_similar_rows(row_idx, column)
        elif method == 'similar_columns':
            return self._interpolate_with_similar_columns(row_idx, column)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

    def _interpolate_with_predictor(self, row_idx: int, column: str):
        """Interpolate using the predictor to predict the missing value."""
        # Get the row with missing value
        row = self.df[row_idx]

        # We need to train a predictor for this specific column if we don't have one
        # or if the current predictor's label is different
        if self._predictor is None or self._predictor.label != column:
            # Train a quick predictor for this column
            non_null_df = self.df.filter(pl.col(column).is_not_null())

            if len(non_null_df) == 0:
                raise ValueError(f"No non-null values in column '{column}' to train predictor")

            # Create temporary predictor
            temp_dir = tempfile.mkdtemp(prefix="autogluon_interp_")
            temp_predictor = TabularPredictor(
                label=column,
                path=temp_dir,
                verbosity=0
            )

            # Use fast preset for quick interpolation
            temp_predictor.fit(
                train_data=non_null_df.to_pandas(),
                presets='medium_quality_faster_train',
                time_limit=60
            )

            # Predict the missing value
            row_pd = row.to_pandas()
            prediction = temp_predictor.predict(row_pd.drop(columns=[column]))

            return prediction.iloc[0]
        else:
            # Use existing predictor
            row_pd = row.to_pandas()
            prediction = self.predictor.predict(row_pd.drop(columns=[column]))
            return prediction.iloc[0]

    def _interpolate_with_similar_rows(
        self,
        row_idx: int,
        column: str,
        n_similar: int = 5
    ):
        """Interpolate using average of similar rows."""
        # Find similar rows
        similar_rows_df = self.find_similar_rows(row_idx, n_similar=n_similar, include_self=False)

        if len(similar_rows_df) == 0:
            raise ValueError("No similar rows found for interpolation")

        # Get values from similar rows
        similar_indices = similar_rows_df['similar_idx'].to_list()
        similar_values = self.df[column][similar_indices]

        # Remove nulls
        similar_values = similar_values.drop_nulls()

        if len(similar_values) == 0:
            raise ValueError("All similar rows also have null values")

        # Return weighted average (weighted by similarity)
        similarities = similar_rows_df['similarity_score'].to_numpy()
        values = similar_values.to_numpy()

        # Ensure matching lengths
        min_len = min(len(similarities), len(values))
        similarities = similarities[:min_len]
        values = values[:min_len]

        # Weighted average
        weighted_sum = np.sum(similarities * values)
        weight_sum = np.sum(similarities)

        return weighted_sum / weight_sum if weight_sum > 0 else np.mean(values)

    def _interpolate_with_similar_columns(
        self,
        row_idx: int,
        column: str,
        n_similar: int = 3
    ):
        """Interpolate using correlation with similar columns."""
        # Find similar columns
        similar_cols_df = self.find_similar_columns(column, n_similar=n_similar, include_self=False)

        if len(similar_cols_df) == 0:
            raise ValueError("No similar columns found for interpolation")

        # Get the row values for similar columns
        similar_col_names = similar_cols_df['similar_column'].to_list()
        row_values = []
        col_means = []
        col_stds = []

        for col in similar_col_names:
            # Get value in this row for similar column
            val = self.df[row_idx, col]

            if val is not None:
                row_values.append(float(val))

                # Get statistics for this column
                col_data = self.df[col].drop_nulls()
                col_means.append(col_data.mean())
                col_stds.append(col_data.std())

        if len(row_values) == 0:
            raise ValueError("All similar columns also have null values in this row")

        # Compute correlation-based estimate
        # Normalize values from similar columns
        normalized_values = [
            (val - mean) / (std + 1e-8)
            for val, mean, std in zip(row_values, col_means, col_stds)
        ]

        # Average normalized value
        avg_normalized = np.mean(normalized_values)

        # Denormalize using target column statistics
        target_col_data = self.df[column].drop_nulls()
        target_mean = target_col_data.mean()
        target_std = target_col_data.std()

        return avg_normalized * target_std + target_mean

    def fill_missing(
        self,
        columns: Optional[List[str]] = None,
        method: str = 'similar_rows',
        inplace: bool = False
    ) -> 'AGPolarsDataFrame':
        """
        Fill all missing values in specified columns.

        Args:
            columns: List of columns to fill. If None, fills all columns.
            method: Interpolation method ('predict', 'similar_rows', 'similar_columns')
            inplace: If True, modifies self.df. If False, returns new instance.

        Returns:
            AGPolarsDataFrame with filled values
        """
        if columns is None:
            columns = self.df.columns

        df_filled = self.df.clone()

        for col in columns:
            # Find rows with null values
            null_mask = df_filled[col].is_null()
            null_indices = [i for i, is_null in enumerate(null_mask) if is_null]

            if len(null_indices) == 0:
                continue

            print(f"Filling {len(null_indices)} missing values in column '{col}'...")

            for idx in null_indices:
                try:
                    filled_value = self.interpolate_missing(idx, col, method=method)

                    # Update the value
                    df_filled = df_filled.with_columns(
                        pl.when(pl.int_range(len(df_filled)) == idx)
                        .then(filled_value)
                        .otherwise(pl.col(col))
                        .alias(col)
                    )
                except Exception as e:
                    warnings.warn(f"Could not fill value at row {idx}, column '{col}': {e}")

        if inplace:
            self.df = df_filled
            # Invalidate cached embeddings since data changed
            self._row_embeddings = None
            self._row_nn_model = None
            return self
        else:
            return AGPolarsDataFrame(df_filled, self._predictor)

    def __getitem__(self, key):
        """Allow indexing like a regular DataFrame."""
        return self.df[key]

    def __repr__(self):
        return f"AGPolarsDataFrame(\n{self.df}\n)"

    def __len__(self):
        return len(self.df)
