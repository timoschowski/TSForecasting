#!/usr/bin/env python
"""
Real comparison of DeepAR with Tweedie vs Normal distribution.
This actually trains both models and compares their forecasts.
"""

import sys
sys.path.insert(0, '/home/user/gluonts/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import torch
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import TweedieOutput, NormalOutput
from gluonts.evaluation import make_evaluation_predictions

print("Setting random seeds...")
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic time series data with Tweedie characteristics
print("\nGenerating synthetic data...")
n_series = 10
n_timesteps = 150
prediction_length = 24
freq = "H"

def generate_tweedie_series(length, base_level=8, amplitude=2.5, period=24, trend=0.015):
    """Generate a time series with Tweedie (compound Poisson-Gamma) noise."""
    t = np.arange(length)

    # True signal
    sinusoid = amplitude * np.sin(2 * np.pi * t / period)
    trend_component = trend * t
    clean_signal = base_level + sinusoid + trend_component

    # Add Tweedie-like noise (compound Poisson-Gamma)
    ts_values = []
    for mu in clean_signal:
        n_events = np.random.poisson(mu * 0.3)
        if n_events == 0:
            ts_values.append(0)
        else:
            gamma_sum = np.random.gamma(2, mu / (2 * n_events), n_events).sum()
            ts_values.append(max(0, gamma_sum))

    return np.array(ts_values)

# Create datasets
train_data = []
test_data = []
start = pd.Timestamp(datetime.now())

for i in range(n_series):
    ts = generate_tweedie_series(n_timesteps + prediction_length)

    train_data.append({
        "start": start,
        "target": ts[:n_timesteps].tolist(),
    })

    test_data.append({
        "start": start,
        "target": ts.tolist(),
    })

train_ds = ListDataset(train_data, freq=freq)
test_ds = ListDataset(test_data, freq=freq)

print(f"Created {n_series} time series")
print(f"  Training length: {n_timesteps}")
print(f"  Prediction length: {prediction_length}")

# Train DeepAR with Normal distribution (default)
print("\n" + "="*70)
print("Training DeepAR with Normal Distribution (default)")
print("="*70)

estimator_normal = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    dropout_rate=0.1,
    distr_output=NormalOutput(),
    batch_size=32,
    num_batches_per_epoch=50,
    trainer_kwargs={
        "max_epochs": 15,
        "accelerator": "cpu",
        "logger": False,
        "enable_progress_bar": True,
    },
)

predictor_normal = estimator_normal.train(train_ds)

print("\nGenerating forecasts (Normal)...")
forecast_it_normal, ts_it_normal = make_evaluation_predictions(
    dataset=test_ds,
    predictor=predictor_normal,
    num_samples=100,
)
forecasts_normal = list(forecast_it_normal)
tss_normal = list(ts_it_normal)

# Train DeepAR with Tweedie distribution
print("\n" + "="*70)
print("Training DeepAR with Tweedie Distribution (p=1.5)")
print("="*70)

estimator_tweedie = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    dropout_rate=0.1,
    distr_output=TweedieOutput(p=1.5),
    batch_size=32,
    num_batches_per_epoch=50,
    trainer_kwargs={
        "max_epochs": 15,
        "accelerator": "cpu",
        "logger": False,
        "enable_progress_bar": True,
    },
)

predictor_tweedie = estimator_tweedie.train(train_ds)

print("\nGenerating forecasts (Tweedie)...")
forecast_it_tweedie, ts_it_tweedie = make_evaluation_predictions(
    dataset=test_ds,
    predictor=predictor_tweedie,
    num_samples=100,
)
forecasts_tweedie = list(forecast_it_tweedie)
tss_tweedie = list(ts_it_tweedie)

# Create comparison plots
print("\n" + "="*70)
print("Creating comparison plots...")
print("="*70)

# Plot 1: Side-by-side forecast comparison
fig, axes = plt.subplots(3, 2, figsize=(18, 12))
fig.suptitle('DeepAR Comparison: Tweedie vs Normal Distribution\n(Real Trained Models on Synthetic Time Series)',
             fontsize=16, fontweight='bold')

for idx in range(min(3, n_series)):
    forecast_normal = forecasts_normal[idx]
    forecast_tweedie = forecasts_tweedie[idx]
    ts = tss_normal[idx]

    # Show last 3*prediction_length points
    display_length = prediction_length * 3
    ts_display = ts[-display_length:]
    hist_length = len(ts_display) - prediction_length

    # Left: Tweedie
    ax_tweedie = axes[idx, 0]
    hist_x = np.arange(hist_length)
    forecast_x = np.arange(hist_length, len(ts_display))

    ax_tweedie.plot(hist_x, ts_display[:hist_length], 'o-', color='black',
                    linewidth=1.5, markersize=3, label='Historical Data', alpha=0.8)
    ax_tweedie.plot(forecast_x, forecast_tweedie.median, color='blue',
                    linewidth=2.5, label='Forecast (Median)', zorder=5)
    ax_tweedie.fill_between(forecast_x, forecast_tweedie.quantile(0.1),
                            forecast_tweedie.quantile(0.9), alpha=0.25,
                            color='blue', label='80% Prediction Interval')
    ax_tweedie.fill_between(forecast_x, forecast_tweedie.quantile(0.25),
                            forecast_tweedie.quantile(0.75), alpha=0.4,
                            color='blue', label='50% Prediction Interval')
    ax_tweedie.plot(forecast_x, ts_display[hist_length:], 'o', color='red',
                    markersize=4, label='Actual Future', alpha=0.7, zorder=6)
    ax_tweedie.axvline(x=hist_length, color='green', linestyle='--',
                       linewidth=2, alpha=0.7)

    ax_tweedie.set_xlabel('Time Step', fontsize=11)
    ax_tweedie.set_ylabel('Value', fontsize=11)
    ax_tweedie.set_title(f'Series {idx+1} - Tweedie (p=1.5)', fontsize=12, fontweight='bold')
    ax_tweedie.legend(loc='best', fontsize=8)
    ax_tweedie.grid(True, alpha=0.3)

    # Calculate metrics
    mae_tweedie = np.mean(np.abs(forecast_tweedie.median - ts_display[hist_length:]))
    zeros_count = (ts_display[:hist_length] == 0).sum()
    stats_text = f'MAE: {mae_tweedie:.2f}\nZeros: {zeros_count}\nVariance: power-law'
    ax_tweedie.text(0.02, 0.98, stats_text, transform=ax_tweedie.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='lightblue', alpha=0.8), fontsize=8)

    # Right: Normal
    ax_normal = axes[idx, 1]
    ax_normal.plot(hist_x, ts_display[:hist_length], 'o-', color='black',
                   linewidth=1.5, markersize=3, label='Historical Data', alpha=0.8)
    ax_normal.plot(forecast_x, forecast_normal.median, color='purple',
                   linewidth=2.5, label='Forecast (Median)', zorder=5)
    ax_normal.fill_between(forecast_x, forecast_normal.quantile(0.1),
                           forecast_normal.quantile(0.9), alpha=0.25,
                           color='purple', label='80% Prediction Interval')
    ax_normal.fill_between(forecast_x, forecast_normal.quantile(0.25),
                           forecast_normal.quantile(0.75), alpha=0.4,
                           color='purple', label='50% Prediction Interval')
    ax_normal.plot(forecast_x, ts_display[hist_length:], 'o', color='red',
                   markersize=4, label='Actual Future', alpha=0.7, zorder=6)
    ax_normal.axvline(x=hist_length, color='green', linestyle='--',
                      linewidth=2, alpha=0.7)

    ax_normal.set_xlabel('Time Step', fontsize=11)
    ax_normal.set_ylabel('Value', fontsize=11)
    ax_normal.set_title(f'Series {idx+1} - Normal (Default)', fontsize=12, fontweight='bold')
    ax_normal.legend(loc='best', fontsize=8)
    ax_normal.grid(True, alpha=0.3)

    mae_normal = np.mean(np.abs(forecast_normal.median - ts_display[hist_length:]))
    stats_text = f'MAE: {mae_normal:.2f}\nZeros: {zeros_count}\nVariance: constant'
    ax_normal.text(0.02, 0.98, stats_text, transform=ax_normal.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='plum', alpha=0.8), fontsize=8)

plt.tight_layout()
plt.savefig('/home/user/TSForecasting/tweedie_deepar_forecast_real.png', dpi=150, bbox_inches='tight')
print("✓ Real forecast comparison saved: tweedie_deepar_forecast_real.png")
plt.close()

# Plot 2: Aggregate comparison metrics
print("\nComputing aggregate metrics...")

# Compute metrics for all series
maes_normal = []
maes_tweedie = []
rmses_normal = []
rmses_tweedie = []

for forecast_normal, forecast_tweedie, ts in zip(forecasts_normal, forecasts_tweedie, tss_normal):
    actuals = ts[-prediction_length:]

    mae_normal = np.mean(np.abs(forecast_normal.median - actuals))
    mae_tweedie = np.mean(np.abs(forecast_tweedie.median - actuals))

    rmse_normal = np.sqrt(np.mean((forecast_normal.median - actuals) ** 2))
    rmse_tweedie = np.sqrt(np.mean((forecast_tweedie.median - actuals) ** 2))

    maes_normal.append(mae_normal)
    maes_tweedie.append(mae_tweedie)
    rmses_normal.append(rmse_normal)
    rmses_tweedie.append(rmse_normal)

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Aggregate Performance Comparison: Tweedie vs Normal', fontsize=14, fontweight='bold')

# MAE comparison
ax = axes2[0]
x_pos = np.arange(2)
means = [np.mean(maes_tweedie), np.mean(maes_normal)]
stds = [np.std(maes_tweedie), np.std(maes_normal)]
colors = ['blue', 'purple']

bars = ax.bar(x_pos, means, yerr=stds, alpha=0.7, color=colors, capsize=10, edgecolor='black')
ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
ax.set_title('MAE Comparison (lower is better)', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Tweedie', 'Normal'])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Improvement percentage
ax = axes2[1]
improvement = ((np.mean(maes_normal) - np.mean(maes_tweedie)) / np.mean(maes_normal)) * 100
colors_imp = ['green' if improvement > 0 else 'red']
bars = ax.bar([0], [improvement], alpha=0.7, color=colors_imp, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel('% Improvement', fontsize=12)
ax.set_title('Tweedie Improvement over Normal', fontsize=12, fontweight='bold')
ax.set_xticks([0])
ax.set_xticklabels(['MAE'])
ax.grid(True, alpha=0.3, axis='y')
ax.text(0, improvement, f'{improvement:+.1f}%', ha='center',
        va='bottom' if improvement > 0 else 'top', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/user/TSForecasting/tweedie_performance_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Performance comparison saved: tweedie_performance_comparison.png")
plt.close()

print("\n" + "="*70)
print("Real comparison completed successfully!")
print("="*70)
print(f"\nAverage MAE - Tweedie: {np.mean(maes_tweedie):.3f} ± {np.std(maes_tweedie):.3f}")
print(f"Average MAE - Normal:  {np.mean(maes_normal):.3f} ± {np.std(maes_normal):.3f}")
print(f"\nImprovement: {improvement:+.1f}%")
print("\nPlots saved:")
print("  - tweedie_deepar_forecast_real.png (actual trained model forecasts)")
print("  - tweedie_performance_comparison.png (aggregate metrics)")
