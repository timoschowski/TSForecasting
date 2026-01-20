#!/usr/bin/env python
"""
Debug Tweedie parameters to understand why median is zero.
"""

import sys
sys.path.insert(0, '/home/user/gluonts/src')

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import TweedieOutput

np.random.seed(42)
torch.manual_seed(42)

# Generate clean sinusoid
freq = "H"
prediction_length = 24
hist_length = 150
n_timesteps = hist_length + prediction_length

t = np.arange(n_timesteps)
scale = 10
ts = scale + (scale * 0.3) * np.sin(2 * np.pi * t / 24)

train_data = [{"target": ts[:hist_length], "start": pd.Timestamp("2021-01-01")}]
test_data = [{"target": ts, "start": pd.Timestamp("2021-01-01")}]

train_ds = ListDataset(train_data, freq=freq)
test_ds = ListDataset(test_data, freq=freq)

# Train Tweedie
estimator_tweedie = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    distr_output=TweedieOutput(p=1.5),
    trainer_kwargs={"max_epochs": 20},
)
predictor_tweedie = estimator_tweedie.train(train_ds)

print("Generating forecast and extracting parameters...")
forecast_tweedie = list(predictor_tweedie.predict(test_ds))[0]

# Access the underlying distribution
# The forecast object contains samples, but we need to get the distribution parameters
print("\nForecast statistics:")
print(f"  Mean: {np.array(forecast_tweedie.mean).mean():.4f}")
print(f"  Median: {np.array(forecast_tweedie.median).mean():.4f}")
print(f"  Q10: {np.array(forecast_tweedie.quantile(0.1)).mean():.4f}")
print(f"  Q90: {np.array(forecast_tweedie.quantile(0.9)).mean():.4f}")

# Try to extract the actual distribution parameters
# We need to run the network forward pass manually
print("\nAttempting to extract learned parameters...")

# Get the network from the predictor
network = predictor_tweedie.prediction_net

# Prepare input
for batch in test_ds:
    target = torch.tensor(batch['target'][:hist_length]).float().unsqueeze(0)
    break

# Run forward pass (simplified - may not work depending on GluonTS version)
try:
    with torch.no_grad():
        # This is a simplified attempt - actual implementation may vary
        output = network(target)
        print(f"\nNetwork output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
except Exception as e:
    print(f"\nCouldn't extract parameters directly: {e}")

# Alternative: Compute P(Y=0) from the compound Poisson-Gamma formula
# For Tweedie 1 < p < 2: P(Y=0) = exp(-lambda) where lambda = mu^(2-p)/(phi*(2-p))
print("\n" + "="*70)
print("THEORETICAL ANALYSIS")
print("="*70)

# Estimate parameters from mean and variance
mean_est = np.array(forecast_tweedie.mean).mean()
# Variance estimate from quantiles
q10 = np.array(forecast_tweedie.quantile(0.1)).mean()
q90 = np.array(forecast_tweedie.quantile(0.9)).mean()
# Rough variance estimate
var_est = ((q90 - q10) / 2.56)**2  # Approximate for wide interval

print(f"\nEstimated parameters:")
print(f"  mu (mean): {mean_est:.4f}")
print(f"  var: {var_est:.4f}")

# For Tweedie: Var = phi * mu^p
# So: phi = Var / mu^p
p = 1.5
if mean_est > 0:
    phi_est = var_est / (mean_est**p)
    print(f"  phi (estimated): {phi_est:.4f}")

    # Compute P(Y=0)
    lambda_param = (mean_est**(2-p)) / (phi_est * (2-p))
    prob_zero = np.exp(-lambda_param)

    print(f"\nCompound Poisson-Gamma parameters:")
    print(f"  lambda (Poisson rate): {lambda_param:.4f}")
    print(f"  P(Y=0): {prob_zero:.4f}")

    if prob_zero > 0.5:
        print(f"\n✗ PROBLEM: P(Y=0) = {prob_zero:.2%} > 50%")
        print("  This means median WILL be zero!")
        print("  The phi parameter is too large relative to mu.")
    else:
        print(f"\n✓ P(Y=0) = {prob_zero:.2%} < 50%")
        print("  Median should be non-zero.")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

print("\nThe issue is that phi is too large, causing:")
print("  1. High dispersion (wide prediction intervals)")
print("  2. High P(Y=0) (median collapses to zero)")
print("\nThis suggests the scaling formula for phi is STILL incorrect,")
print("OR there's another issue with how phi is learned during training.")
