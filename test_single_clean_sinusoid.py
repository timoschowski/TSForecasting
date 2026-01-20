#!/usr/bin/env python
"""
Test Tweedie on a single clean sinusoidal curve with no noise.
This should yield nearly perfect MASE values for both distributions.
"""

import sys
sys.path.insert(0, '/home/user/gluonts/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    mase = mae_forecast / mae_naive
    return mase

print("Setting random seeds...")
np.random.seed(42)
torch.manual_seed(42)

# Generate a single clean sinusoidal series (NO NOISE)
freq = "H"
prediction_length = 24
hist_length = 150
n_timesteps = hist_length + prediction_length

print("\nGenerating perfectly clean sinusoidal series...")
t = np.arange(n_timesteps)
# Pure sinusoid with no noise at all
scale = 10
ts = scale + (scale * 0.3) * np.sin(2 * np.pi * t / 24)

print(f"  Training length: {hist_length}")
print(f"  Prediction length: {prediction_length}")
print(f"  Scale: {scale}")
print(f"  Value range: [{ts.min():.2f}, {ts.max():.2f}]")

# Prepare datasets
train_data_normal = [{
    "target": ts[:hist_length],
    "start": pd.Timestamp("2021-01-01")
}]
train_data_tweedie = [{
    "target": ts[:hist_length],
    "start": pd.Timestamp("2021-01-01")
}]
test_data = [{
    "target": ts,
    "start": pd.Timestamp("2021-01-01")
}]

train_ds_normal = ListDataset(train_data_normal, freq=freq)
train_ds_tweedie = ListDataset(train_data_tweedie, freq=freq)
test_ds = ListDataset(test_data, freq=freq)

# Train Normal model
print("\n" + "="*70)
print("Training DeepAR with Normal Distribution")
print("="*70)
estimator_normal = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    distr_output=NormalOutput(),
    trainer_kwargs={"max_epochs": 20},
)
predictor_normal = estimator_normal.train(train_ds_normal)

# Train Tweedie model
print("\n" + "="*70)
print("Training DeepAR with Tweedie Distribution (p=1.5)")
print("="*70)
estimator_tweedie = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    distr_output=TweedieOutput(p=1.5),
    trainer_kwargs={"max_epochs": 20},
)
predictor_tweedie = estimator_tweedie.train(train_ds_tweedie)

# Generate forecasts
print("\nGenerating forecasts...")
forecast_normal = list(predictor_normal.predict(test_ds))[0]
forecast_tweedie = list(predictor_tweedie.predict(test_ds))[0]

# Compute metrics
print("\n" + "="*70)
print("RESULTS")
print("="*70)

actuals = ts[hist_length:]
training = ts[:hist_length]

median_normal = np.array(forecast_normal.median)
median_tweedie = np.array(forecast_tweedie.median)

mase_normal = compute_mase(median_normal, actuals, training)
mase_tweedie = compute_mase(median_tweedie, actuals, training)

# Compute additional metrics
mae_normal = np.mean(np.abs(median_normal - actuals))
mae_tweedie = np.mean(np.abs(median_tweedie - actuals))

rmse_normal = np.sqrt(np.mean((median_normal - actuals)**2))
rmse_tweedie = np.sqrt(np.mean((median_tweedie - actuals)**2))

print(f"\nActual values range: [{actuals.min():.4f}, {actuals.max():.4f}]")
print(f"\n{'Distribution':<12} {'MASE':<10} {'MAE':<10} {'RMSE':<10} {'Median Range'}")
print("-" * 70)
print(f"{'Normal':<12} {mase_normal:<10.6f} {mae_normal:<10.6f} {rmse_normal:<10.6f} [{median_normal.min():.4f}, {median_normal.max():.4f}]")
print(f"{'Tweedie':<12} {mase_tweedie:<10.6f} {mae_tweedie:<10.6f} {rmse_tweedie:<10.6f} [{median_tweedie.min():.4f}, {median_tweedie.max():.4f}]")

print(f"\n{'='*70}")
print("COMPARISON")
print('='*70)
if mase_tweedie < mase_normal:
    improvement = ((mase_normal - mase_tweedie) / mase_normal) * 100
    print(f"Tweedie is {improvement:.2f}% better than Normal")
elif mase_normal < mase_tweedie:
    improvement = ((mase_tweedie - mase_normal) / mase_tweedie) * 100
    print(f"Normal is {improvement:.2f}% better than Tweedie")
else:
    print("Both distributions perform equally")

print(f"\nExpectation: Both should have near-perfect MASE (close to 0.0)")
print(f"Both MASE values < 0.1? {mase_normal < 0.1 and mase_tweedie < 0.1}")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('DeepAR on Single Clean Sinusoid: Tweedie vs Normal',
             fontsize=14, fontweight='bold')

hist_x = np.arange(hist_length)
forecast_x = np.arange(hist_length, hist_length + prediction_length)

# Left: Tweedie
ax_tweedie = axes[0]
ax_tweedie.plot(hist_x, ts[:hist_length], 'o-', color='black',
               linewidth=1.5, markersize=3, label='Historical Data', alpha=0.8)
ax_tweedie.plot(forecast_x, forecast_tweedie.median, color='blue',
               linewidth=2.5, label='Forecast (Median)', zorder=5)
ax_tweedie.fill_between(forecast_x, forecast_tweedie.quantile(0.1),
                        forecast_tweedie.quantile(0.9), alpha=0.25,
                        color='blue', label='80% Prediction Interval')
ax_tweedie.fill_between(forecast_x, forecast_tweedie.quantile(0.25),
                        forecast_tweedie.quantile(0.75), alpha=0.4,
                        color='blue', label='50% Prediction Interval')
ax_tweedie.plot(forecast_x, ts[hist_length:], 'o', color='red',
               markersize=5, label='Actual Future', alpha=0.7, zorder=6)
ax_tweedie.axvline(x=hist_length, color='green', linestyle='--',
                   linewidth=2, alpha=0.7)

ax_tweedie.set_xlabel('Time Step', fontsize=12)
ax_tweedie.set_ylabel('Value', fontsize=12)
ax_tweedie.set_title('Tweedie Distribution (p=1.5)',
                    fontsize=13, fontweight='bold')
ax_tweedie.legend(loc='best', fontsize=9)
ax_tweedie.grid(True, alpha=0.3)

stats_text = f'MASE: {mase_tweedie:.6f}\nMAE: {mae_tweedie:.6f}\nRMSE: {rmse_tweedie:.6f}\nPerfectly clean sinusoid'
ax_tweedie.text(0.02, 0.98, stats_text, transform=ax_tweedie.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='lightblue', alpha=0.8), fontsize=9)

# Right: Normal
ax_normal = axes[1]
ax_normal.plot(hist_x, ts[:hist_length], 'o-', color='black',
               linewidth=1.5, markersize=3, label='Historical Data', alpha=0.8)
ax_normal.plot(forecast_x, forecast_normal.median, color='purple',
               linewidth=2.5, label='Forecast (Median)', zorder=5)
ax_normal.fill_between(forecast_x, forecast_normal.quantile(0.1),
                       forecast_normal.quantile(0.9), alpha=0.25,
                       color='purple', label='80% Prediction Interval')
ax_normal.fill_between(forecast_x, forecast_normal.quantile(0.25),
                       forecast_normal.quantile(0.75), alpha=0.4,
                       color='purple', label='50% Prediction Interval')
ax_normal.plot(forecast_x, ts[hist_length:], 'o', color='red',
               markersize=5, label='Actual Future', alpha=0.7, zorder=6)
ax_normal.axvline(x=hist_length, color='green', linestyle='--',
                  linewidth=2, alpha=0.7)

ax_normal.set_xlabel('Time Step', fontsize=12)
ax_normal.set_ylabel('Value', fontsize=12)
ax_normal.set_title('Normal Distribution',
                   fontsize=13, fontweight='bold')
ax_normal.legend(loc='best', fontsize=9)
ax_normal.grid(True, alpha=0.3)

stats_text = f'MASE: {mase_normal:.6f}\nMAE: {mae_normal:.6f}\nRMSE: {rmse_normal:.6f}\nPerfectly clean sinusoid'
ax_normal.text(0.02, 0.98, stats_text, transform=ax_normal.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='plum', alpha=0.8), fontsize=9)

plt.tight_layout()
plt.savefig('/home/user/TSForecasting/single_clean_sinusoid_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved: single_clean_sinusoid_comparison.png")
