# AutoGluon-Enhanced Polars DataFrame

This module provides functionality to enhance Polars DataFrames with AutoGluon's tabular foundation models, enabling:

- **Row similarity search** using embeddings from foundation models
- **Column similarity search** using statistical embeddings
- **Missing value interpolation** using foundation models and similarity-based methods

## Features

### 1. Row Similarity Search
Find similar rows in your DataFrame using embeddings derived from AutoGluon's foundation models or feature transformations.

```python
# Find 5 rows most similar to row 10
similar_rows = ag_df.find_similar_rows(row_indices=10, n_similar=5)
```

### 2. Column Similarity Search
Find similar columns based on statistical properties and correlations.

```python
# Find 3 columns most similar to 'temperature'
similar_cols = ag_df.find_similar_columns(column_names='temperature', n_similar=3)
```

### 3. Missing Value Interpolation
Interpolate missing values using multiple methods:
- **similar_rows**: Uses weighted average of similar rows
- **similar_columns**: Uses correlation with similar columns
- **predict**: Uses AutoGluon predictor to predict missing values (requires trained model)

```python
# Interpolate a single missing value
value = ag_df.interpolate_missing(row_idx=5, column='temperature', method='similar_rows')

# Fill all missing values in the DataFrame
ag_df_filled = ag_df.fill_missing(method='similar_rows')
```

## Installation

### Required Dependencies

```bash
pip install polars pandas numpy scikit-learn
pip install autogluon.tabular
```

Or install from the requirements file:

```bash
pip install -r requirements_ag_polars.txt
```

### Note on AutoGluon
AutoGluon's tabular module includes several foundation models:
- **Mitra**: Tabular foundation model pre-trained on synthetic data
- **TabPFN**: In-context learning for tabular data
- **TabICL**: Tabular in-context learning
- And more...

When using `presets='best_quality'`, AutoGluon will automatically leverage these foundation models.

## Usage

### Basic Usage (Without Training a Predictor)

```python
import polars as pl
from utils.ag_polars_dataframe import AGPolarsDataFrame

# Load or create your data
df = pl.DataFrame({
    'temperature': [20.5, 21.0, None, 22.5, 19.8],
    'humidity': [65, 70, 68, None, 72],
    'pressure': [1013, 1012, 1014, 1013, None],
})

# Create AGPolarsDataFrame
ag_df = AGPolarsDataFrame(df)

# Find similar rows
similar = ag_df.find_similar_rows(row_indices=0, n_similar=3)
print(similar)

# Find similar columns
similar_cols = ag_df.find_similar_columns(column_names='temperature', n_similar=2)
print(similar_cols)

# Interpolate missing values
ag_df_filled = ag_df.fill_missing(method='similar_rows')
print(ag_df_filled.df)
```

### Advanced Usage (With Trained Predictor)

```python
import polars as pl
from utils.ag_polars_dataframe import AGPolarsDataFrame

# Load your data
df = pl.read_csv('your_data.csv')

# Create AGPolarsDataFrame and train a predictor
ag_df = AGPolarsDataFrame(df)

# Train on a target variable
ag_df.fit_predictor(
    label='target_column',
    presets='best_quality',  # Uses foundation models
    time_limit=300  # 5 minutes
)

# Now you can use prediction-based interpolation
ag_df_filled = ag_df.fill_missing(method='predict')
```

## API Reference

### AGPolarsDataFrame

#### Constructor
```python
AGPolarsDataFrame(df: pl.DataFrame, predictor: Optional[TabularPredictor] = None)
```

#### Methods

##### fit_predictor
Train a TabularPredictor on the DataFrame.

```python
fit_predictor(
    label: str,
    presets: str = "best_quality",
    time_limit: Optional[int] = None,
    **kwargs
) -> AGPolarsDataFrame
```

**Parameters:**
- `label`: Column name to predict
- `presets`: Quality preset ('best_quality', 'high_quality', 'medium_quality_faster_train')
- `time_limit`: Training time limit in seconds
- `**kwargs`: Additional arguments for TabularPredictor.fit()

##### find_similar_rows
Find similar rows using embeddings.

```python
find_similar_rows(
    row_indices: Union[int, List[int]],
    n_similar: int = 5,
    include_self: bool = False
) -> pl.DataFrame
```

**Parameters:**
- `row_indices`: Single index or list of indices
- `n_similar`: Number of similar rows to return
- `include_self`: Whether to include query row in results

**Returns:** DataFrame with columns `query_idx`, `similar_idx`, `similarity_score`

##### find_similar_columns
Find similar columns using statistical embeddings.

```python
find_similar_columns(
    column_names: Union[str, List[str]],
    n_similar: int = 3,
    include_self: bool = False
) -> pl.DataFrame
```

**Parameters:**
- `column_names`: Single column or list of columns
- `n_similar`: Number of similar columns to return
- `include_self`: Whether to include query column in results

**Returns:** DataFrame with columns `query_column`, `similar_column`, `similarity_score`

##### interpolate_missing
Interpolate a single missing value.

```python
interpolate_missing(
    row_idx: int,
    column: str,
    method: str = 'predict'
) -> Union[float, int, str]
```

**Parameters:**
- `row_idx`: Row index with missing value
- `column`: Column name with missing value
- `method`: Interpolation method ('predict', 'similar_rows', 'similar_columns')

**Returns:** Interpolated value

##### fill_missing
Fill all missing values in specified columns.

```python
fill_missing(
    columns: Optional[List[str]] = None,
    method: str = 'similar_rows',
    inplace: bool = False
) -> AGPolarsDataFrame
```

**Parameters:**
- `columns`: List of columns to fill (None = all columns)
- `method`: Interpolation method
- `inplace`: If True, modifies self.df; if False, returns new instance

**Returns:** AGPolarsDataFrame with filled values

## Running the Demo

A comprehensive demo script is provided that demonstrates all features:

```bash
# First, ensure all dependencies are installed
pip install -r requirements_ag_polars.txt

# Run the demo
python demo_ag_polars.py
```

The demo will:
1. Create a synthetic weather/energy dataset
2. Demonstrate row similarity search
3. Demonstrate column similarity search
4. Show different missing value interpolation methods
5. Fill all missing values
6. Optionally train a predictor for prediction-based interpolation

## Implementation Details

### Embeddings

**Row Embeddings:**
- If a predictor is trained, uses `predictor.transform_features()` to get feature representations
- Falls back to numeric columns if predictor is not available
- Uses cosine similarity for finding similar rows

**Column Embeddings:**
- Computes statistical features for each column: mean, std, min, max, median, null count, non-null count
- Uses these statistics as embeddings for column similarity

### Missing Value Interpolation Methods

1. **similar_rows**:
   - Finds N most similar rows using embeddings
   - Computes weighted average based on similarity scores
   - Fast and works without training a model

2. **similar_columns**:
   - Finds columns most similar to the target column
   - Uses correlation patterns to estimate the missing value
   - Normalizes values across columns for better estimates

3. **predict**:
   - Trains or uses an existing AutoGluon predictor
   - Predicts the missing value based on other features
   - Most accurate but requires training time

## Integration with Polars Operations

The `AGPolarsDataFrame` wraps a Polars DataFrame and preserves access to the underlying data:

```python
# Access the underlying Polars DataFrame
polars_df = ag_df.df

# Use regular Polars operations
filtered_df = ag_df.df.filter(pl.col('temperature') > 20)

# Create new AGPolarsDataFrame from filtered data
ag_df_filtered = AGPolarsDataFrame(filtered_df, ag_df._predictor)
```

## Performance Considerations

- **Embedding computation**: Cached after first computation (recomputed if data changes)
- **Nearest neighbors**: Uses sklearn's NearestNeighbors with cosine metric
- **Missing value interpolation**: Predictor-based method is slower but more accurate
- **Memory**: Embeddings are stored in memory; consider batch processing for very large datasets

## Future Enhancements

Possible future improvements:
1. Override Polars operations directly (e.g., custom `filter()`, `select()`)
2. Support for categorical columns in similarity search
3. Batch interpolation optimization
4. Custom embedding models
5. Streaming support for large datasets
6. Integration with AutoGluon's MultiModalPredictor for text/image columns

## Sources and References

- [AutoGluon Documentation](https://auto.gluon.ai/stable/index.html)
- [AutoGluon Tabular Foundation Models](https://auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html)
- [MultiModalPredictor.extract_embedding](https://auto.gluon.ai/stable/api/autogluon.multimodal.MultiModalPredictor.extract_embedding.html)
- [TabularPredictor.transform_features](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.transform_features.html)
- [Polars Documentation](https://pola-rs.github.io/polars/)

## License

This code is part of the TSForecasting repository. Please refer to the main repository license.
