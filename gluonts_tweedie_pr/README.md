# Tweedie Distribution Implementation for GluonTS

This directory contains a complete implementation of the Tweedie distribution for GluonTS DeepAR, ready to be submitted as a pull request to the awslabs/gluonts repository.

## Files in this Directory

1. **tweedie.py** - Core implementation
   - Place at: `src/gluonts/torch/distributions/tweedie.py`

2. **test_tweedie.py** - Comprehensive unit tests
   - Place at: `test/torch/distribution/test_tweedie.py`

3. **tweedie_deepar_demo.py** - Full demonstration script
   - Place at: `examples/tweedie_deepar_demo.py`

4. **verify_tweedie.py** - Quick verification script
   - Place at root of gluonts repository

5. **__init__.py.patch** - Changes needed for distributions __init__.py
   - Apply to: `src/gluonts/torch/distributions/__init__.py`

6. **TWEEDIE_IMPLEMENTATION.md** - Detailed documentation
   - Place at root of gluonts repository

## How to Create the Pull Request

### Step 1: Fork the Repository

Fork https://github.com/awslabs/gluonts to your GitHub account.

### Step 2: Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/gluonts.git
cd gluonts
git checkout dev  # Work from the dev branch
```

### Step 3: Create Feature Branch

```bash
git checkout -b add-tweedie-distribution
```

### Step 4: Add the Files

```bash
# Copy the implementation
cp /path/to/tweedie.py src/gluonts/torch/distributions/

# Copy the tests
cp /path/to/test_tweedie.py test/torch/distribution/

# Copy the demo
cp /path/to/tweedie_deepar_demo.py examples/

# Copy documentation
cp /path/to/TWEEDIE_IMPLEMENTATION.md .
cp /path/to/verify_tweedie.py .

# Apply the init patch
cd src/gluonts/torch/distributions/
patch < /path/to/__init__.py.patch
cd -
```

Or manually update `src/gluonts/torch/distributions/__init__.py`:

Add import:
```python
from .tweedie import Tweedie, TweedieOutput
```

Add to __all__:
```python
"Tweedie",
"TweedieOutput",
```

### Step 5: Test the Implementation

```bash
# Install gluonts in development mode
pip install -e .

# Run the tests
python -m pytest test/torch/distribution/test_tweedie.py -v

# Run quick verification
python verify_tweedie.py

# Run the demo (optional)
python examples/tweedie_deepar_demo.py
```

### Step 6: Commit the Changes

```bash
git add src/gluonts/torch/distributions/tweedie.py
git add src/gluonts/torch/distributions/__init__.py
git add test/torch/distribution/test_tweedie.py
git add examples/tweedie_deepar_demo.py
git add TWEEDIE_IMPLEMENTATION.md
git add verify_tweedie.py

git commit -m "Add Tweedie distribution support for DeepAR

This commit implements the Tweedie distribution (compound Poisson-Gamma)
for probabilistic forecasting in GluonTS, particularly useful for modeling
time series with zero-inflation and positive continuous values.

Key features:
- Tweedie distribution class with full PyTorch distributions API
- TweedieOutput for seamless integration with DeepAR and other models
- Support for sampling from the distribution via compound Poisson-Gamma
- Proper parameter constraints and domain mapping
- Handles zero-inflated data naturally

Files added/modified:
- src/gluonts/torch/distributions/tweedie.py: Core implementation
- src/gluonts/torch/distributions/__init__.py: Export Tweedie classes
- test/torch/distribution/test_tweedie.py: Comprehensive unit tests
- examples/tweedie_deepar_demo.py: Full demonstration with synthetic data
- TWEEDIE_IMPLEMENTATION.md: Detailed documentation
- verify_tweedie.py: Quick verification script

Typical use cases:
- Insurance claims modeling
- Rainfall prediction
- Web traffic analytics
- Any intermittent event processes"
```

### Step 7: Push to Your Fork

```bash
git push -u origin add-tweedie-distribution
```

### Step 8: Create Pull Request

1. Go to https://github.com/YOUR_USERNAME/gluonts
2. Click "New Pull Request"
3. Select base repository: `awslabs/gluonts` base: `dev`
4. Select head repository: `YOUR_USERNAME/gluonts` compare: `add-tweedie-distribution`
5. Fill in the PR template with:

```markdown
## Description

This PR implements the Tweedie distribution for probabilistic forecasting in GluonTS.  The Tweedie distribution (specifically the compound Poisson-Gamma case for 1 < p < 2) is particularly useful for modeling time series with:
- Zero-inflation (many zero values)
- Positive continuous values
- Over-dispersion

## Use Cases

- Insurance claims modeling
- Rainfall prediction
- Web traffic analytics with intermittent spikes
- Any process with intermittent events of varying magnitude

## Implementation Details

### Core Features

- **Tweedie Distribution Class**: Full implementation extending `torch.distributions.Distribution`
- **TweedieOutput Class**: Seamless integration with DeepAR and other GluonTS models
- **Sampling Support**: Uses compound Poisson-Gamma representation for generating samples
- **Parameter Constraints**: Automatic domain mapping ensures valid parameters
- **Zero-Inflation Handling**: Exact computation of P(Y=0) using Poisson component

### Mathematical Background

The Tweedie distribution with power parameter p ∈ (1, 2) is a compound Poisson-Gamma distribution:
- Mean: E[Y] = μ
- Variance: Var[Y] = φ × μ^p

Where:
- μ: mean parameter (positive)
- φ: dispersion parameter (positive)
- p: power parameter (default 1.5)

## Files Added/Modified

- `src/gluonts/torch/distributions/tweedie.py`: Core implementation
- `src/gluonts/torch/distributions/__init__.py`: Exports for Tweedie classes
- `test/torch/distribution/test_tweedie.py`: Comprehensive unit tests
- `examples/tweedie_deepar_demo.py`: Complete demonstration with synthetic data
- `TWEEDIE_IMPLEMENTATION.md`: Detailed documentation
- `verify_tweedie.py`: Quick verification script

## Testing

The implementation includes:
- Unit tests for distribution properties (mean, variance)
- Sampling tests
- Log probability computation tests
- Parameter validation tests
- Domain mapping tests
- Integration tests with TweedieOutput

Run tests with:
```bash
python -m pytest test/torch/distribution/test_tweedie.py -v
```

## Example Usage

```python
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import TweedieOutput

# Create Tweedie distribution output
distr_output = TweedieOutput(p=1.5)

# Use in DeepAR
estimator = DeepAREstimator(
    freq="H",
    prediction_length=24,
    distr_output=distr_output,
    trainer_kwargs={"max_epochs": 20},
)

predictor = estimator.train(train_dataset)
forecasts = predictor.predict(test_dataset)
```

## Demonstration

A complete demonstration script is included (`examples/tweedie_deepar_demo.py`) that:
1. Generates synthetic time series with sinusoidal patterns, noise, and linear trend
2. Trains a DeepAR model with Tweedie distribution
3. Generates probabilistic forecasts with prediction intervals
4. Creates visualization plots
5. Computes evaluation metrics

Run with:
```bash
python examples/tweedie_deepar_demo.py
```

## References

- Tweedie, M.C.K. (1984). "An index which distinguishes between some important exponential families"
- Jørgensen, B. (1987). "Exponential dispersion models"
- Dunn, P.K. and Smyth, G.K. (2005). "Series evaluation of Tweedie exponential dispersion model densities"

## Checklist

- [x] Implementation follows GluonTS patterns and conventions
- [x] Comprehensive unit tests included
- [x] Documentation and examples provided
- [x] Code follows PEP 8 style guidelines
- [x] All tests pass locally
- [x] Distribution properly integrates with DeepAR
- [x] Sampling support implemented
- [x] Parameter constraints handled correctly
```

## Expected Review Points

The reviewers may ask about:

1. **Exact PDF computation**: Current implementation uses an approximation suitable for optimization. For exact PDF, Wright's generalized Bessel function would be needed (available in scipy.special.wright_bessel since scipy 1.7.0).

2. **CDF/Inverse CDF**: Not currently implemented. Could be added in future enhancement.

3. **Special cases**: Currently optimized for 1 < p < 2. Special cases at p=0 (Normal), p=1 (Poisson), p=2 (Gamma), p=3 (Inverse Gaussian) could be added.

4. **Sampling efficiency**: Current implementation uses an approximation. More sophisticated methods could improve accuracy for large batches.

## Related Issues/Discussions

- This addresses use cases similar to those discussed in time series forecasting for intermittent data
- Complements existing distribution options (Normal, StudentT, NegativeBinomial, etc.)
- Particularly useful for demand forecasting with many zeros

## License

All code follows the Apache License 2.0 as required by the GluonTS project.
