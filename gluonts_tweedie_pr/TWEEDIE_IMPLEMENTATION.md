# Tweedie Distribution Implementation for GluonTS

This document describes the implementation of the Tweedie distribution for DeepAR in GluonTS.

## Overview

The Tweedie distribution is a member of the exponential dispersion family that is particularly useful for modeling data with:
- Many zero values
- Positive continuous values
- Over-dispersion

For the power parameter `1 < p < 2`, the Tweedie distribution represents a compound Poisson-Gamma distribution, making it ideal for applications such as:
- Insurance claims modeling
- Rainfall prediction
- Web traffic analytics
- Any process with intermittent events of varying magnitude

## Implementation Details

### Files Modified/Created

1. **`src/gluonts/torch/distributions/tweedie.py`** (NEW)
   - `Tweedie`: PyTorch distribution class extending `torch.distributions.Distribution`
   - `TweedieOutput`: Distribution output class for GluonTS integration

2. **`src/gluonts/torch/distributions/__init__.py`** (MODIFIED)
   - Added exports for `Tweedie` and `TweedieOutput`

3. **`test/torch/distribution/test_tweedie.py`** (NEW)
   - Comprehensive unit tests for the Tweedie distribution
   - Tests for sampling, log probability, parameter validation, etc.

4. **`examples/tweedie_deepar_demo.py`** (NEW)
   - Complete demonstration with synthetic data
   - Shows sinusoidal time series with noise and trend
   - Trains DeepAR model and generates forecasts

## Mathematical Background

### Parameters

The Tweedie distribution has three parameters:

- **μ (mu)**: Mean parameter (must be positive)
- **φ (phi)**: Dispersion parameter (must be positive)
- **p**: Power parameter (must be in (1, 2) for compound Poisson-Gamma case)

### Properties

- **Mean**: E[Y] = μ
- **Variance**: Var[Y] = φ × μ^p

### Compound Poisson-Gamma Representation

For 1 < p < 2, the Tweedie distribution can be represented as:

Y = Σ(i=1 to N) Z_i

where:
- N ~ Poisson(λ)
- Z_i ~ Gamma(α, β)
- λ = μ^(2-p) / (φ × (2-p))
- α = (2-p) / (p-1)
- β = μ^(1-p) / (φ × (p-1))

## Usage

### Basic Usage with DeepAR

```python
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import TweedieOutput

# Create Tweedie distribution output with p=1.5
distr_output = TweedieOutput(p=1.5)

# Use in DeepAR estimator
estimator = DeepAREstimator(
    freq="H",
    prediction_length=24,
    num_layers=2,
    hidden_size=40,
    distr_output=distr_output,
    trainer_kwargs={"max_epochs": 20},
)

# Train and predict
predictor = estimator.train(train_dataset)
forecasts = predictor.predict(test_dataset)
```

### Direct Distribution Usage

```python
import torch
from gluonts.torch.distributions import Tweedie

# Create distribution
mu = torch.tensor([1.0, 2.0, 3.0])
phi = torch.tensor([0.5, 1.0, 1.5])
p = 1.5

dist = Tweedie(mu=mu, phi=phi, p=p)

# Get properties
print(f"Mean: {dist.mean}")
print(f"Variance: {dist.variance}")

# Sample
samples = dist.sample((1000,))

# Compute log probability
values = torch.tensor([0.0, 1.0, 2.0])
log_probs = dist.log_prob(values)
```

## Testing

### Run Unit Tests

```bash
cd /home/user/gluonts
python -m pytest test/torch/distribution/test_tweedie.py -v
```

### Run Quick Verification

```bash
python verify_tweedie.py
```

### Run Full Demonstration

```bash
python examples/tweedie_deepar_demo.py
```

This will:
1. Generate synthetic time series with sinusoidal patterns, noise, and trend
2. Train a DeepAR model with Tweedie distribution
3. Generate probabilistic forecasts
4. Create visualization plots
5. Print evaluation metrics

## Key Features

### Fully Probabilistic

- Provides complete probability distribution, not just point forecasts
- Supports quantile forecasts and prediction intervals
- Handles uncertainty properly

### Sampling Support

- Implements `sample()` method for generating random samples
- Uses compound Poisson-Gamma representation for sampling
- Enables Monte Carlo-based forecasting

### Proper Parameter Constraints

- Automatic domain mapping ensures parameters stay in valid ranges
- μ and φ constrained to be positive via softplus transformation
- p constrained to (1, 2) for compound Poisson-Gamma case

### Zero-Inflation Handling

- Naturally handles datasets with many zero values
- Exact computation of P(Y=0) using Poisson component
- Suitable for intermittent or sparse time series

## Implementation Notes

### Log-Likelihood Computation

The implementation uses the canonical parameterization of the Tweedie distribution:

```
θ = μ^(1-p) / (1-p)
κ(θ) = μ^(2-p) / (2-p)
log L = (y×θ - κ(θ)) / φ
```

For zero values, the exact probability is computed using:
```
P(Y=0) = exp(-λ) where λ = μ^(2-p) / (φ×(2-p))
```

### Sampling Strategy

The sampling uses an efficient approximation of the compound Poisson-Gamma representation:
1. Sample number of events from Poisson distribution
2. Sample event magnitudes from Gamma distribution
3. Combine to generate final samples

### Limitations

- Currently optimized for 1 < p < 2 (compound Poisson-Gamma case)
- Does not implement rsample (reparameterization trick) due to discrete component
- Log-likelihood uses approximation suitable for optimization (not exact PDF)

## References

1. Tweedie, M.C.K. (1984). "An index which distinguishes between some important exponential families"
2. Jørgensen, B. (1987). "Exponential dispersion models"
3. Dunn, P.K. and Smyth, G.K. (2005). "Series evaluation of Tweedie exponential dispersion model densities"

## Future Enhancements

Potential improvements for future versions:

1. Exact PDF computation using Wright's generalized Bessel function (available in scipy.special)
2. CDF and inverse CDF (quantile function) implementations
3. Support for other special cases (p=0: Normal, p=1: Poisson, p=2: Gamma, p=3: Inverse Gaussian)
4. More efficient sampling for large batch sizes
5. GPU-optimized implementations

## License

Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0.
