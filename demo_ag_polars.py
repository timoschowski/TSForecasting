"""
Demo script for AutoGluon-Enhanced Polars DataFrame

This script demonstrates:
1. Creating an AGPolarsDataFrame
2. Finding similar rows using embeddings
3. Finding similar columns using embeddings
4. Interpolating missing values using foundation models
"""

import polars as pl
import numpy as np
from utils.ag_polars_dataframe import AGPolarsDataFrame

# Set random seed for reproducibility
np.random.seed(42)


def create_sample_data():
    """Create a sample dataset with some patterns and missing values."""
    print("=" * 80)
    print("Creating sample dataset...")
    print("=" * 80)

    # Create a synthetic dataset with patterns
    n_rows = 100

    # Feature 1: Temperature (Celsius)
    temperature = np.random.normal(20, 5, n_rows)

    # Feature 2: Humidity (%) - correlated with temperature
    humidity = 60 - 0.5 * temperature + np.random.normal(0, 5, n_rows)

    # Feature 3: Pressure (hPa) - weakly correlated
    pressure = np.random.normal(1013, 10, n_rows)

    # Feature 4: Wind Speed (km/h)
    wind_speed = np.random.exponential(15, n_rows)

    # Feature 5: Rainfall (mm) - related to humidity
    rainfall = np.maximum(0, (humidity - 50) / 10 + np.random.exponential(2, n_rows))

    # Target: Energy Consumption (kWh) - depends on temperature and other factors
    energy = (
        100
        + 2 * temperature
        - 0.3 * humidity
        + 0.5 * wind_speed
        + np.random.normal(0, 10, n_rows)
    )

    # Create DataFrame
    df = pl.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'rainfall': rainfall,
        'energy_consumption': energy
    })

    # Introduce some missing values (10% missing rate)
    n_missing = int(0.1 * n_rows)

    # Add missing values to different columns
    missing_indices_temp = np.random.choice(n_rows, n_missing, replace=False)
    missing_indices_humidity = np.random.choice(n_rows, n_missing, replace=False)
    missing_indices_rainfall = np.random.choice(n_rows, n_missing, replace=False)

    # Create masks for missing values
    temp_mask = np.ones(n_rows, dtype=bool)
    temp_mask[missing_indices_temp] = False

    humidity_mask = np.ones(n_rows, dtype=bool)
    humidity_mask[missing_indices_humidity] = False

    rainfall_mask = np.ones(n_rows, dtype=bool)
    rainfall_mask[missing_indices_rainfall] = False

    # Apply missing values
    df = df.with_columns([
        pl.when(pl.lit(temp_mask)).then(pl.col('temperature')).alias('temperature'),
        pl.when(pl.lit(humidity_mask)).then(pl.col('humidity')).alias('humidity'),
        pl.when(pl.lit(rainfall_mask)).then(pl.col('rainfall')).alias('rainfall'),
    ])

    print(f"\nDataset created with {n_rows} rows and {len(df.columns)} columns")
    print(f"Missing values: {df.null_count().sum_horizontal()[0]} total")
    print("\nFirst 5 rows:")
    print(df.head())

    return df


def demo_row_similarity(ag_df: AGPolarsDataFrame):
    """Demonstrate finding similar rows."""
    print("\n" + "=" * 80)
    print("DEMO 1: Finding Similar Rows")
    print("=" * 80)

    # Select a few sample rows
    sample_indices = [0, 25, 50]

    print(f"\nFinding rows similar to indices: {sample_indices}")
    print("\nOriginal rows:")
    for idx in sample_indices:
        print(f"\nRow {idx}:")
        print(ag_df.df[idx])

    # Find similar rows
    similar_rows = ag_df.find_similar_rows(
        row_indices=sample_indices,
        n_similar=3,
        include_self=False
    )

    print("\n" + "-" * 80)
    print("Similar rows found:")
    print(similar_rows)

    # Show details of similar rows for first query
    if len(similar_rows) > 0:
        first_query = sample_indices[0]
        similar_to_first = similar_rows.filter(pl.col('query_idx') == first_query)

        print(f"\n\nDetailed comparison for Row {first_query}:")
        print(f"Query row {first_query}:")
        print(ag_df.df[first_query])

        for row in similar_to_first.iter_rows(named=True):
            sim_idx = row['similar_idx']
            sim_score = row['similarity_score']
            print(f"\nSimilar row {sim_idx} (similarity: {sim_score:.4f}):")
            print(ag_df.df[sim_idx])


def demo_column_similarity(ag_df: AGPolarsDataFrame):
    """Demonstrate finding similar columns."""
    print("\n" + "=" * 80)
    print("DEMO 2: Finding Similar Columns")
    print("=" * 80)

    # Find columns similar to temperature
    query_columns = ['temperature', 'humidity']

    print(f"\nFinding columns similar to: {query_columns}")

    similar_cols = ag_df.find_similar_columns(
        column_names=query_columns,
        n_similar=3,
        include_self=False
    )

    print("\n" + "-" * 80)
    print("Similar columns found:")
    print(similar_cols)

    # Show statistics for comparison
    print("\n\nColumn statistics for comparison:")
    for col in ag_df.df.columns:
        col_data = ag_df.df[col].drop_nulls()
        if len(col_data) > 0:
            print(f"\n{col}:")
            print(f"  Mean: {col_data.mean():.2f}")
            print(f"  Std:  {col_data.std():.2f}")
            print(f"  Min:  {col_data.min():.2f}")
            print(f"  Max:  {col_data.max():.2f}")


def demo_missing_value_interpolation(ag_df: AGPolarsDataFrame):
    """Demonstrate missing value interpolation."""
    print("\n" + "=" * 80)
    print("DEMO 3: Missing Value Interpolation")
    print("=" * 80)

    # Find a row with missing temperature
    missing_temp_idx = None
    for i, val in enumerate(ag_df.df['temperature']):
        if val is None:
            missing_temp_idx = i
            break

    if missing_temp_idx is None:
        print("\nNo missing values found in temperature column")
        return

    print(f"\nInterpolating missing temperature at row {missing_temp_idx}")
    print(f"Original row:")
    print(ag_df.df[missing_temp_idx])

    # Try different interpolation methods
    methods = ['similar_rows', 'similar_columns']

    for method in methods:
        print(f"\n" + "-" * 80)
        print(f"Using method: {method}")

        try:
            interpolated_value = ag_df.interpolate_missing(
                row_idx=missing_temp_idx,
                column='temperature',
                method=method
            )

            print(f"Interpolated temperature: {interpolated_value:.2f}°C")

            # Show similar rows or columns used
            if method == 'similar_rows':
                similar = ag_df.find_similar_rows(missing_temp_idx, n_similar=5)
                print(f"\nBased on {len(similar)} similar rows")
            elif method == 'similar_columns':
                similar = ag_df.find_similar_columns('temperature', n_similar=3)
                print(f"\nBased on {len(similar)} similar columns:")
                print(similar)

        except Exception as e:
            print(f"Error with method {method}: {e}")


def demo_fill_all_missing(ag_df: AGPolarsDataFrame):
    """Demonstrate filling all missing values."""
    print("\n" + "=" * 80)
    print("DEMO 4: Fill All Missing Values")
    print("=" * 80)

    # Count missing values before
    missing_before = ag_df.df.null_count()
    print("\nMissing values before interpolation:")
    print(missing_before)

    # Fill missing values
    print("\nFilling missing values using 'similar_rows' method...")
    ag_df_filled = ag_df.fill_missing(method='similar_rows', inplace=False)

    # Count missing values after
    missing_after = ag_df_filled.df.null_count()
    print("\nMissing values after interpolation:")
    print(missing_after)

    # Show comparison
    print("\n" + "-" * 80)
    print("Summary:")
    total_before = missing_before.sum_horizontal()[0]
    total_after = missing_after.sum_horizontal()[0]
    print(f"Total missing values: {total_before} -> {total_after}")
    print(f"Values filled: {total_before - total_after}")


def main():
    """Main demo function."""
    print("\n" + "=" * 80)
    print("AutoGluon-Enhanced Polars DataFrame Demo")
    print("=" * 80)

    # Create sample data
    df = create_sample_data()

    # Create AGPolarsDataFrame (without training a predictor first)
    # The embeddings will be computed from the data features
    print("\n" + "=" * 80)
    print("Creating AGPolarsDataFrame...")
    print("=" * 80)

    ag_df = AGPolarsDataFrame(df)

    # Note: We're not training a predictor here because we're just demonstrating
    # the similarity and interpolation features that work with embeddings
    # derived from the data itself.

    # Demo 1: Row similarity
    demo_row_similarity(ag_df)

    # Demo 2: Column similarity
    demo_column_similarity(ag_df)

    # Demo 3: Missing value interpolation
    demo_missing_value_interpolation(ag_df)

    # Demo 4: Fill all missing values
    demo_fill_all_missing(ag_df)

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)

    # Optional: Train a predictor for prediction-based interpolation
    print("\n" + "=" * 80)
    print("BONUS: Training a predictor for prediction-based interpolation")
    print("=" * 80)
    print("\nNote: This may take a few minutes...")

    try:
        # Create a version without missing values for training
        df_train = df.drop_nulls()

        if len(df_train) > 20:  # Need enough data
            ag_df_with_predictor = AGPolarsDataFrame(df_train)

            # Train on a subset of data with short time limit
            print("\nTraining predictor on 'energy_consumption'...")
            ag_df_with_predictor.fit_predictor(
                label='energy_consumption',
                presets='medium_quality_faster_train',
                time_limit=120  # 2 minutes
            )

            print("\n✓ Predictor trained successfully!")
            print("You can now use method='predict' for interpolation.")

            # Try prediction-based interpolation on original data with missing values
            ag_df_with_pred = AGPolarsDataFrame(df, ag_df_with_predictor._predictor)

            # Find a missing value in a non-target column
            for col in ['temperature', 'humidity', 'rainfall']:
                for i, val in enumerate(ag_df_with_pred.df[col]):
                    if val is None:
                        print(f"\nTrying prediction-based interpolation for {col} at row {i}")
                        try:
                            pred_val = ag_df_with_pred.interpolate_missing(
                                row_idx=i,
                                column=col,
                                method='predict'
                            )
                            print(f"Predicted value: {pred_val:.2f}")
                        except Exception as e:
                            print(f"Note: {e}")
                        break
                break

    except Exception as e:
        print(f"\nNote: Could not train predictor: {e}")
        print("This is optional - the main functionality works without it.")

    print("\n" + "=" * 80)
    print("All demos completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
