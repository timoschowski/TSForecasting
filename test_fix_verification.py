#!/usr/bin/env python
"""
Quick verification that the scaling fix works.
Train fresh models with the corrected code.
"""

import sys
sys.path.insert(0, '/home/user/gluonts/src')

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import TweedieOutput, NormalOutput

def compute_mase(forecast, actuals, training_series):
    mae_forecast = np.mean(np.abs(forecast - actuals))
    naive_errors = np.abs(np.diff(training_series))
    mae_naive = np.mean(naive_errors)
    if mae_naive == 0 or np.isnan(mae_naive) or np.isinf(mae_naive):
        mae_naive = 1e-10
    return mae_forecast / mae_naive

print("="*70)
print("TESTING TWEEDIE SCALING FIX")
print("="*70)

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

print(f"\nData: Clean sinusoid, range [{ts.min():.2f}, {ts.max():.2f}]")

# Prepare datasets
train_data = [{"target": ts[:hist_length], "start": pd.Timestamp("2021-01-01")}]
test_data = [{"target": ts, "start": pd.Timestamp("2021-01-01")}]

train_ds = ListDataset(train_data, freq=freq)
test_ds = ListDataset(test_data, freq=freq)

# Train Tweedie model with FIXED code
print("\n" + "="*70)
print("Training Tweedie (with corrected scaling formula)")
print("="*70)

estimator_tweedie = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    distr_output=TweedieOutput(p=1.5),
    trainer_kwargs={"max_epochs": 20},
)
predictor_tweedie = estimator_tweedie.train(train_ds)

print("\nGenerating Tweedie forecast...")
forecast_tweedie = list(predictor_tweedie.predict(test_ds))[0]

actuals = ts[hist_length:]
training = ts[:hist_length]

median_tweedie = np.array(forecast_tweedie.median)
mean_tweedie = np.array(forecast_tweedie.mean)

print("\n" + "="*70)
print("RESULTS")
print("="*70)

print(f"\nActuals: [{actuals.min():.2f}, {actuals.max():.2f}]")
print(f"Tweedie median: [{median_tweedie.min():.2f}, {median_tweedie.max():.2f}]")
print(f"Tweedie mean: [{mean_tweedie.min():.2f}, {mean_tweedie.max():.2f}]")

mase_tweedie = compute_mase(median_tweedie, actuals, training)
mae_tweedie = np.mean(np.abs(median_tweedie - actuals))

print(f"\nTweedie MASE: {mase_tweedie:.4f}")
print(f"Tweedie MAE: {mae_tweedie:.4f}")

print("\n" + "="*70)
if mase_tweedie < 5.0:
    print("✓ SUCCESS: Tweedie MASE is reasonable!")
    print("  The scaling fix appears to be working.")
else:
    print("✗ ISSUE: Tweedie MASE is still too high.")
    print("  Further investigation needed.")

print(f"\nExpected: MASE should be around 2-3 (similar to Normal)")
print(f"Got: MASE = {mase_tweedie:.4f}")
