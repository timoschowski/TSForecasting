#!/usr/bin/env python
"""
Test PyTorch-style Tweedie (with full log-likelihood) vs Normal on clean sinusoid.

This implements the CORRECT Tweedie formula with normalizing constant via
series expansion, following PyTorch PR #171705 approach.

This should fix the phi gradient bug and achieve performance similar to Normal.
"""

import sys
sys.path.insert(0, '/home/user/gluonts/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import NormalOutput
from gluonts.torch.distributions.tweedie_full import TweedieFullOutput
from gluonts.torch.distributions import TweedieOutput  # Old deviance-based

def compute_mase(forecast, actuals, training_series):
    """Mean Absolute Scaled Error"""
    mae_forecast = np.mean(np.abs(forecast - actuals))
    naive_errors = np.abs(np.diff(training_series))
    mae_naive = np.mean(naive_errors)
    if mae_naive == 0 or np.isnan(mae_naive) or np.isinf(mae_naive):
        mae_naive = 1e-10
    mase = mae_forecast / mae_naive
    return mase

print("="*80)
print("PYTORCH-STYLE TWEEDIE (FULL LOG-LIKELIHOOD) VS NORMAL")
print("="*80)
print("\nThis tests whether implementing the full log-likelihood (with normalizing")
print("constant via series expansion) fixes the phi gradient bug.")

# Set random seeds
print("\nSetting random seeds...")
np.random.seed(42)
torch.manual_seed(42)

# Generate clean sinusoidal series
freq = "H"
prediction_length = 24
hist_length = 150
n_timesteps = hist_length + prediction_length

print("\nGenerating perfectly clean sinusoidal series...")
t = np.arange(n_timesteps)
scale = 10
ts = scale + (scale * 0.3) * np.sin(2 * np.pi * t / 24)

print(f"  Training length: {hist_length}")
print(f"  Prediction length: {prediction_length}")
print(f"  Value range: [{ts.min():.2f}, {ts.max():.2f}]")

# Prepare datasets
train_data = [{"target": ts[:hist_length], "start": pd.Timestamp("2021-01-01")}]
test_data = [{"target": ts, "start": pd.Timestamp("2021-01-01")}]

train_ds = ListDataset(train_data, freq=freq)
test_ds = ListDataset(test_data, freq=freq)

# Train Normal model (baseline)
print("\n" + "="*80)
print("1. Training DeepAR with Normal Distribution (Baseline)")
print("="*80)
estimator_normal = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    distr_output=NormalOutput(),
    trainer_kwargs={"max_epochs": 20},
)
predictor_normal = estimator_normal.train(train_ds)

# Train Tweedie model with OLD deviance-based formula (for comparison)
print("\n" + "="*80)
print("2. Training DeepAR with Tweedie (OLD: Deviance-based)")
print("="*80)
print("This is the broken implementation with incorrect phi gradients...")
np.random.seed(42)
torch.manual_seed(42)

estimator_tweedie_old = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    distr_output=TweedieOutput(p=1.5),
    trainer_kwargs={"max_epochs": 20},
)
predictor_tweedie_old = estimator_tweedie_old.train(train_ds)

# Train Tweedie model with NEW full log-likelihood (PyTorch-style)
print("\n" + "="*80)
print("3. Training DeepAR with TweedieFull (NEW: Full log-likelihood)")
print("="*80)
print("This uses series expansion for normalizing constant (correct gradients)...")
np.random.seed(42)
torch.manual_seed(42)

estimator_tweedie_full = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    distr_output=TweedieFullOutput(p=1.5),
    trainer_kwargs={"max_epochs": 20},
)
predictor_tweedie_full = estimator_tweedie_full.train(train_ds)

# Generate forecasts
print("\nGenerating forecasts...")
forecast_normal = list(predictor_normal.predict(test_ds))[0]
forecast_tweedie_old = list(predictor_tweedie_old.predict(test_ds))[0]
forecast_tweedie_full = list(predictor_tweedie_full.predict(test_ds))[0]

# Extract medians
median_normal = np.array(forecast_normal.median)
median_tweedie_old = np.array(forecast_tweedie_old.median)
median_tweedie_full = np.array(forecast_tweedie_full.median)

# Compute metrics
actuals = ts[hist_length:]
training = ts[:hist_length]

mase_normal = compute_mase(median_normal, actuals, training)
mase_tweedie_old = compute_mase(median_tweedie_old, actuals, training)
mase_tweedie_full = compute_mase(median_tweedie_full, actuals, training)

mae_normal = np.mean(np.abs(median_normal - actuals))
mae_tweedie_old = np.mean(np.abs(median_tweedie_old - actuals))
mae_tweedie_full = np.mean(np.abs(median_tweedie_full - actuals))

rmse_normal = np.sqrt(np.mean((median_normal - actuals)**2))
rmse_tweedie_old = np.sqrt(np.mean((median_tweedie_old - actuals)**2))
rmse_tweedie_full = np.sqrt(np.mean((median_tweedie_full - actuals)**2))

# Compute prediction interval coverage
q10_normal = np.array(forecast_normal.quantile(0.1))
q90_normal = np.array(forecast_normal.quantile(0.9))
q10_tweedie_old = np.array(forecast_tweedie_old.quantile(0.1))
q90_tweedie_old = np.array(forecast_tweedie_old.quantile(0.9))
q10_tweedie_full = np.array(forecast_tweedie_full.quantile(0.1))
q90_tweedie_full = np.array(forecast_tweedie_full.quantile(0.9))

coverage_normal = np.sum((actuals >= q10_normal) & (actuals <= q90_normal)) / len(actuals) * 100
coverage_tweedie_old = np.sum((actuals >= q10_tweedie_old) & (actuals <= q90_tweedie_old)) / len(actuals) * 100
coverage_tweedie_full = np.sum((actuals >= q10_tweedie_full) & (actuals <= q90_tweedie_full)) / len(actuals) * 100

# Print results
print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nActual values range: [{actuals.min():.4f}, {actuals.max():.4f}]")
print(f"\n{'Approach':<30} {'MASE':<12} {'MAE':<12} {'RMSE':<12} {'80% Coverage'}")
print("-" * 85)
print(f"{'Normal (baseline)':<30} {mase_normal:<12.6f} {mae_normal:<12.6f} {rmse_normal:<12.6f} {coverage_normal:>6.1f}%")
print(f"{'Tweedie OLD (deviance)':<30} {mase_tweedie_old:<12.6f} {mae_tweedie_old:<12.6f} {rmse_tweedie_old:<12.6f} {coverage_tweedie_old:>6.1f}%")
print(f"{'TweedieFull NEW (PyTorch)':<30} {mase_tweedie_full:<12.6f} {mae_tweedie_full:<12.6f} {rmse_tweedie_full:<12.6f} {coverage_tweedie_full:>6.1f}%")

# Analysis
print(f"\n{'='*80}")
print("ANALYSIS")
print('='*80)

pct_diff_old = (mase_tweedie_old - mase_normal) / mase_normal * 100
pct_diff_full = (mase_tweedie_full - mase_normal) / mase_normal * 100

print(f"\nMASE Comparison vs Normal:")
print(f"  Normal (baseline):      {mase_normal:.6f}")
print(f"  Tweedie OLD (deviance): {mase_tweedie_old:.6f} ({pct_diff_old:+.1f}%)")
print(f"  TweedieFull NEW:        {mase_tweedie_full:.6f} ({pct_diff_full:+.1f}%)")

print(f"\nKey Question: Does full log-likelihood fix the gradient bug?")
if abs(pct_diff_full) < 10:
    print(f"  ✅ YES! TweedieFull ({pct_diff_full:+.1f}%) matches Normal (within 10%)")
    print(f"  → Implementing correct gradients fixes the issue!")
    print(f"  → Gradient bug was the primary problem")
elif abs(pct_diff_full) < abs(pct_diff_old) and abs(pct_diff_full) < 30:
    print(f"  ✓ IMPROVEMENT! TweedieFull ({pct_diff_full:+.1f}%) is better than OLD ({pct_diff_old:+.1f}%)")
    print(f"  → Full log-likelihood helps significantly")
    print(f"  → Still room for improvement (within 30% of Normal)")
elif abs(pct_diff_full) < abs(pct_diff_old):
    print(f"  → PARTIAL. TweedieFull ({pct_diff_full:+.1f}%) is better than OLD ({pct_diff_old:+.1f}%), but not close to Normal")
    print(f"  → Other issues may exist beyond gradient bug")
else:
    print(f"  ❌ NO. TweedieFull ({pct_diff_full:+.1f}%) doesn't improve over OLD ({pct_diff_old:+.1f}%)")
    print(f"  → May be implementation issues or other problems")

# Forecast range comparison
print(f"\nForecast Ranges:")
print(f"  Normal:      [{median_normal.min():.2f}, {median_normal.max():.2f}]")
print(f"  Tweedie OLD: [{median_tweedie_old.min():.2f}, {median_tweedie_old.max():.2f}]")
print(f"  Tweedie NEW: [{median_tweedie_full.min():.2f}, {median_tweedie_full.max():.2f}]")
print(f"  Actual:      [{actuals.min():.2f}, {actuals.max():.2f}]")

# Create visualization
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

hist_x = np.arange(hist_length)
forecast_x = np.arange(hist_length, hist_length + prediction_length)

# Top row: Normal and Tweedie OLD
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(hist_x, ts[:hist_length], 'o-', color='black', linewidth=1.5,
         markersize=2, label='Historical', alpha=0.7)
ax1.plot(forecast_x, median_normal, color='purple', linewidth=2.5,
         label='Forecast', zorder=5)
ax1.fill_between(forecast_x, q10_normal, q90_normal, alpha=0.3, color='purple')
ax1.plot(forecast_x, actuals, 'o', color='red', markersize=4,
         label='Actual', alpha=0.8, zorder=6)
ax1.axvline(x=hist_length, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax1.set_xlabel('Time Step', fontsize=11)
ax1.set_ylabel('Value', fontsize=11)
ax1.set_title('Normal (Baseline)', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
stats_text = f'MASE: {mase_normal:.4f}\nCoverage: {coverage_normal:.1f}%'
ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8), fontsize=9)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(hist_x, ts[:hist_length], 'o-', color='black', linewidth=1.5,
         markersize=2, label='Historical', alpha=0.7)
ax2.plot(forecast_x, median_tweedie_old, color='darkorange', linewidth=2.5,
         label='Forecast', zorder=5)
ax2.fill_between(forecast_x, q10_tweedie_old, q90_tweedie_old, alpha=0.3, color='darkorange')
ax2.plot(forecast_x, actuals, 'o', color='red', markersize=4,
         label='Actual', alpha=0.8, zorder=6)
ax2.axvline(x=hist_length, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_xlabel('Time Step', fontsize=11)
ax2.set_ylabel('Value', fontsize=11)
ax2.set_title('Tweedie OLD (Deviance-based)', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
stats_text = f'MASE: {mase_tweedie_old:.4f} ({pct_diff_old:+.1f}%)\nCoverage: {coverage_tweedie_old:.1f}%'
ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.8), fontsize=9)

# Middle: Tweedie NEW (full log-likelihood)
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(hist_x, ts[:hist_length], 'o-', color='black', linewidth=1.5,
         markersize=2, label='Historical', alpha=0.7)
ax3.plot(forecast_x, median_tweedie_full, color='green', linewidth=2.5,
         label='Forecast', zorder=5)
ax3.fill_between(forecast_x, q10_tweedie_full, q90_tweedie_full, alpha=0.3, color='green')
ax3.plot(forecast_x, actuals, 'o', color='red', markersize=4,
         label='Actual', alpha=0.8, zorder=6)
ax3.axvline(x=hist_length, color='darkgreen', linestyle='--', linewidth=2, alpha=0.5)
ax3.set_xlabel('Time Step', fontsize=11)
ax3.set_ylabel('Value', fontsize=11)
ax3.set_title('TweedieFull NEW (PyTorch-style: Full log-likelihood with series expansion)',
              fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)
stats_text = f'MASE: {mase_tweedie_full:.4f} ({pct_diff_full:+.1f}%)\nCoverage: {coverage_tweedie_full:.1f}%'
ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=9)

# Bottom: Direct comparison
ax4 = fig.add_subplot(gs[2, :])
ax4.plot(forecast_x, actuals, 'o-', color='red', linewidth=2.5, markersize=5,
         label='Actual', zorder=10, alpha=0.8)
ax4.plot(forecast_x, median_normal, 's-', color='purple', linewidth=2,
         markersize=4, label='Normal', alpha=0.7)
ax4.plot(forecast_x, median_tweedie_old, '^-', color='darkorange', linewidth=2,
         markersize=4, label='Tweedie OLD', alpha=0.7)
ax4.plot(forecast_x, median_tweedie_full, 'D-', color='green', linewidth=2,
         markersize=4, label='TweedieFull NEW', alpha=0.7)
ax4.set_xlabel('Time Step', fontsize=11)
ax4.set_ylabel('Value', fontsize=11)
ax4.set_title('Direct Comparison: All Forecasts', fontsize=12, fontweight='bold')
ax4.legend(loc='best', fontsize=10)
ax4.grid(True, alpha=0.3)

fig.suptitle(f'PyTorch-Style Tweedie (Full Log-Likelihood) Test | Improvement: {pct_diff_old - pct_diff_full:.1f}%',
             fontsize=14, fontweight='bold', y=0.995)

plt.savefig('/home/user/TSForecasting/pytorch_style_tweedie_test.png',
            dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved: pytorch_style_tweedie_test.png")

# Conclusion
print(f"\n{'='*80}")
print("CONCLUSION")
print('='*80)

if abs(pct_diff_full) < 10:
    print(f"🎉 SUCCESS! TweedieFull (PyTorch-style) achieves {pct_diff_full:+.1f}% vs Normal!")
    print(f"   The full log-likelihood with correct gradients FIXES the gradient bug.")
    print(f"   This validates that the mathematical error was the root cause.")
    print(f"\n   Recommendation: Use TweedieFull for Tweedie forecasting in GluonTS")
elif abs(pct_diff_full) < abs(pct_diff_old):
    improvement = pct_diff_old - pct_diff_full
    print(f"✓ IMPROVEMENT! TweedieFull is {improvement:.1f}% points better than OLD.")
    print(f"   OLD: {pct_diff_old:+.1f}% vs Normal")
    print(f"   NEW: {pct_diff_full:+.1f}% vs Normal")
    print(f"\n   Full log-likelihood helps, but may need further tuning.")
    print(f"   Consider: learning rate adjustments, more epochs, hyperparameter search")
else:
    print(f"⚠️ Unexpected: TweedieFull ({pct_diff_full:+.1f}%) not better than OLD ({pct_diff_old:+.1f}%)")
    print(f"   May have implementation issues to debug.")
    print(f"   Check: series bounds, numerical stability, scaling interactions")

print(f"\nNote: Training with series expansion is slower (~10-50x) but mathematically correct.")
