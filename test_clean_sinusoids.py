#!/usr/bin/env python
"""
Test Tweedie on 3 clean sinusoidal curves at different scales.
This tests if the parameter scaling works correctly across different magnitudes.
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

# Generate 3 clean sinusoidal series at different scales
freq = "H"
prediction_length = 24
hist_length = 150
n_timesteps = hist_length + prediction_length

# Three scales: small, medium, large
scales = [5, 10, 20]
series_data = []

for scale in scales:
    t = np.arange(n_timesteps)
    # Pure sinusoid at different scales
    ts = scale + (scale * 0.3) * np.sin(2 * np.pi * t / 24)
    series_data.append(ts)

print(f"\nGenerated {len(scales)} clean sinusoidal series at scales: {scales}")
print(f"  Training length: {hist_length}")
print(f"  Prediction length: {prediction_length}")

# Prepare datasets
train_data_normal = []
train_data_tweedie = []
test_data = []

for ts in series_data:
    train_data_normal.append({
        "target": ts[:hist_length],
        "start": pd.Timestamp("2021-01-01")
    })
    train_data_tweedie.append({
        "target": ts[:hist_length],
        "start": pd.Timestamp("2021-01-01")
    })
    test_data.append({
        "target": ts,
        "start": pd.Timestamp("2021-01-01")
    })

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
    trainer_kwargs={"max_epochs": 15},
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
    trainer_kwargs={"max_epochs": 15},
)
predictor_tweedie = estimator_tweedie.train(train_ds_tweedie)

# Generate forecasts
print("\nGenerating forecasts...")
forecasts_normal = list(predictor_normal.predict(test_ds))
forecasts_tweedie = list(predictor_tweedie.predict(test_ds))

# Compute metrics
print("\n" + "="*70)
print("RESULTS")
print("="*70)

mases_normal = []
mases_tweedie = []

for i, (scale, ts, forecast_normal, forecast_tweedie) in enumerate(
    zip(scales, series_data, forecasts_normal, forecasts_tweedie)
):
    actuals = ts[hist_length:]
    training = ts[:hist_length]

    median_normal = np.array(forecast_normal.median)
    median_tweedie = np.array(forecast_tweedie.median)

    mase_normal = compute_mase(median_normal, actuals, training)
    mase_tweedie = compute_mase(median_tweedie, actuals, training)

    mases_normal.append(mase_normal)
    mases_tweedie.append(mase_tweedie)

    print(f"\nSeries {i+1} (scale={scale}):")
    print(f"  Actual range: [{actuals.min():.2f}, {actuals.max():.2f}]")
    print(f"  Normal  - Median range: [{median_normal.min():.2f}, {median_normal.max():.2f}], MASE: {mase_normal:.3f}")
    print(f"  Tweedie - Median range: [{median_tweedie.min():.2f}, {median_tweedie.max():.2f}], MASE: {mase_tweedie:.3f}")

print(f"\n{'='*70}")
print("AGGREGATE METRICS")
print('='*70)
print(f"Normal  MASE: {np.mean(mases_normal):.3f} ± {np.std(mases_normal):.3f}")
print(f"Tweedie MASE: {np.mean(mases_tweedie):.3f} ± {np.std(mases_tweedie):.3f}")
improvement = ((np.mean(mases_normal) - np.mean(mases_tweedie)) / np.mean(mases_normal)) * 100
print(f"Improvement: {improvement:+.1f}%")

# Create visualization
fig, axes = plt.subplots(len(scales), 2, figsize=(14, 4*len(scales)))
fig.suptitle('DeepAR Comparison: Tweedie vs Normal on Clean Sinusoids',
             fontsize=14, fontweight='bold')

for idx, (scale, ts, forecast_normal, forecast_tweedie) in enumerate(
    zip(scales, series_data, forecasts_normal, forecasts_tweedie)
):
    hist_x = np.arange(hist_length)
    forecast_x = np.arange(hist_length, hist_length + prediction_length)

    # Left: Tweedie
    ax_tweedie = axes[idx, 0]
    ax_tweedie.plot(hist_x, ts[:hist_length], 'o-', color='black',
                   linewidth=1.5, markersize=2, label='Historical Data', alpha=0.8)
    ax_tweedie.plot(forecast_x, forecast_tweedie.median, color='blue',
                   linewidth=2.5, label='Forecast (Median)', zorder=5)
    ax_tweedie.fill_between(forecast_x, forecast_tweedie.quantile(0.1),
                            forecast_tweedie.quantile(0.9), alpha=0.25,
                            color='blue', label='80% Prediction Interval')
    ax_tweedie.fill_between(forecast_x, forecast_tweedie.quantile(0.25),
                            forecast_tweedie.quantile(0.75), alpha=0.4,
                            color='blue', label='50% Prediction Interval')
    ax_tweedie.plot(forecast_x, ts[hist_length:], 'o', color='red',
                   markersize=4, label='Actual Future', alpha=0.7, zorder=6)
    ax_tweedie.axvline(x=hist_length, color='green', linestyle='--',
                       linewidth=2, alpha=0.7)

    ax_tweedie.set_xlabel('Time Step', fontsize=11)
    ax_tweedie.set_ylabel('Value', fontsize=11)
    ax_tweedie.set_title(f'Series {idx+1} - Tweedie (scale={scale})',
                        fontsize=12, fontweight='bold')
    ax_tweedie.legend(loc='best', fontsize=8)
    ax_tweedie.grid(True, alpha=0.3)

    actuals = ts[hist_length:]
    training = ts[:hist_length]
    mase_tweedie = compute_mase(np.array(forecast_tweedie.median), actuals, training)
    stats_text = f'MASE: {mase_tweedie:.3f}\nClean sinusoid'
    ax_tweedie.text(0.02, 0.98, stats_text, transform=ax_tweedie.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='lightblue', alpha=0.8), fontsize=8)

    # Right: Normal
    ax_normal = axes[idx, 1]
    ax_normal.plot(hist_x, ts[:hist_length], 'o-', color='black',
                   linewidth=1.5, markersize=2, label='Historical Data', alpha=0.8)
    ax_normal.plot(forecast_x, forecast_normal.median, color='purple',
                   linewidth=2.5, label='Forecast (Median)', zorder=5)
    ax_normal.fill_between(forecast_x, forecast_normal.quantile(0.1),
                           forecast_normal.quantile(0.9), alpha=0.25,
                           color='purple', label='80% Prediction Interval')
    ax_normal.fill_between(forecast_x, forecast_normal.quantile(0.25),
                           forecast_normal.quantile(0.75), alpha=0.4,
                           color='purple', label='50% Prediction Interval')
    ax_normal.plot(forecast_x, ts[hist_length:], 'o', color='red',
                   markersize=4, label='Actual Future', alpha=0.7, zorder=6)
    ax_normal.axvline(x=hist_length, color='green', linestyle='--',
                      linewidth=2, alpha=0.7)

    ax_normal.set_xlabel('Time Step', fontsize=11)
    ax_normal.set_ylabel('Value', fontsize=11)
    ax_normal.set_title(f'Series {idx+1} - Normal (scale={scale})',
                       fontsize=12, fontweight='bold')
    ax_normal.legend(loc='best', fontsize=8)
    ax_normal.grid(True, alpha=0.3)

    mase_normal = compute_mase(np.array(forecast_normal.median), actuals, training)
    stats_text = f'MASE: {mase_normal:.3f}\nClean sinusoid'
    ax_normal.text(0.02, 0.98, stats_text, transform=ax_normal.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='plum', alpha=0.8), fontsize=8)

plt.tight_layout()
plt.savefig('/home/user/TSForecasting/clean_sinusoid_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved: clean_sinusoid_comparison.png")
