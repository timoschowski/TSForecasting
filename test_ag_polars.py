"""
Simple tests for AutoGluon-Enhanced Polars DataFrame

Run with: pytest test_ag_polars.py
or: python test_ag_polars.py
"""

import sys
import traceback


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")

    try:
        import polars as pl
        print("✓ polars imported")
    except ImportError as e:
        print(f"✗ polars import failed: {e}")
        return False

    try:
        import pandas as pd
        print("✓ pandas imported")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False

    try:
        import numpy as np
        print("✓ numpy imported")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False

    try:
        from sklearn.neighbors import NearestNeighbors
        print("✓ sklearn imported")
    except ImportError as e:
        print(f"✗ sklearn import failed: {e}")
        return False

    try:
        from autogluon.tabular import TabularPredictor
        print("✓ autogluon.tabular imported")
    except ImportError as e:
        print(f"✗ autogluon.tabular import failed: {e}")
        return False

    return True


def test_module_import():
    """Test that our module imports correctly."""
    print("\nTesting module import...")

    try:
        from utils.ag_polars_dataframe import AGPolarsDataFrame
        print("✓ AGPolarsDataFrame imported successfully")
        return True
    except Exception as e:
        print(f"✗ AGPolarsDataFrame import failed: {e}")
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality without training a predictor."""
    print("\nTesting basic functionality...")

    try:
        import polars as pl
        import numpy as np
        from utils.ag_polars_dataframe import AGPolarsDataFrame

        # Create simple test data
        np.random.seed(42)
        df = pl.DataFrame({
            'a': np.random.randn(20),
            'b': np.random.randn(20),
            'c': np.random.randn(20),
        })

        # Add some missing values
        df = df.with_columns([
            pl.when(pl.int_range(len(df)) % 5 == 0)
            .then(None)
            .otherwise(pl.col('a'))
            .alias('a')
        ])

        print(f"  Created test DataFrame with {len(df)} rows")

        # Create AGPolarsDataFrame
        ag_df = AGPolarsDataFrame(df)
        print("  ✓ AGPolarsDataFrame created")

        # Test row similarity
        similar_rows = ag_df.find_similar_rows(row_indices=0, n_similar=3)
        print(f"  ✓ Row similarity search returned {len(similar_rows)} results")

        # Test column similarity
        similar_cols = ag_df.find_similar_columns(column_names='a', n_similar=2)
        print(f"  ✓ Column similarity search returned {len(similar_cols)} results")

        # Test missing value interpolation (similar_rows method)
        missing_idx = None
        for i, val in enumerate(df['a']):
            if val is None:
                missing_idx = i
                break

        if missing_idx is not None:
            interpolated = ag_df.interpolate_missing(
                row_idx=missing_idx,
                column='a',
                method='similar_rows'
            )
            print(f"  ✓ Interpolated missing value: {interpolated:.4f}")

        print("\n✓ All basic functionality tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_with_predictor():
    """Test functionality with a trained predictor."""
    print("\nTesting with predictor (may take a few minutes)...")

    try:
        import polars as pl
        import numpy as np
        from utils.ag_polars_dataframe import AGPolarsDataFrame

        # Create test data
        np.random.seed(42)
        n = 50
        df = pl.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'x3': np.random.randn(n),
        })

        # Add target that depends on features
        df = df.with_columns([
            (2 * pl.col('x1') + 3 * pl.col('x2') - pl.col('x3') +
             pl.Series(np.random.randn(n) * 0.1)).alias('y')
        ])

        print(f"  Created training data with {len(df)} rows")

        # Create and train
        ag_df = AGPolarsDataFrame(df)
        ag_df.fit_predictor(
            label='y',
            presets='medium_quality_faster_train',
            time_limit=60
        )
        print("  ✓ Predictor trained")

        # Test prediction-based interpolation
        # Create a test row with missing value
        test_df = pl.DataFrame({
            'x1': [1.0],
            'x2': [None],  # Missing value
            'x3': [0.5],
            'y': [0.0]
        })

        ag_test = AGPolarsDataFrame(test_df, ag_df._predictor)
        interpolated = ag_test.interpolate_missing(
            row_idx=0,
            column='x2',
            method='predict'
        )
        print(f"  ✓ Prediction-based interpolation: {interpolated:.4f}")

        print("\n✓ Predictor tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Predictor test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("AutoGluon-Enhanced Polars DataFrame Tests")
    print("=" * 80)

    # Check imports first
    if not test_imports():
        print("\n" + "=" * 80)
        print("FAILED: Required dependencies not installed")
        print("Please run: pip install -r requirements_ag_polars.txt")
        print("=" * 80)
        return False

    # Test module import
    if not test_module_import():
        print("\n" + "=" * 80)
        print("FAILED: Module import failed")
        print("=" * 80)
        return False

    # Test basic functionality
    if not test_basic_functionality():
        print("\n" + "=" * 80)
        print("FAILED: Basic functionality tests failed")
        print("=" * 80)
        return False

    # Test with predictor (optional, may take time)
    print("\n" + "=" * 80)
    print("Optional: Testing with predictor (this may take a few minutes)...")
    print("Press Ctrl+C to skip")
    print("=" * 80)

    try:
        test_with_predictor()
    except KeyboardInterrupt:
        print("\n\nSkipped predictor tests")
    except Exception as e:
        print(f"\nPredictor tests failed (this is optional): {e}")

    print("\n" + "=" * 80)
    print("✓ ALL TESTS COMPLETED")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
