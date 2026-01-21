#!/usr/bin/env python
"""
Experiment: Fix Tweedie phi to Normal's learned parameters.

Strategy:
1. Train Normal DeepAR and get its learned parameters (mu, std)
2. Convert Normal parameters to Tweedie: phi = var / mu^p
3. Train Tweedie DeepAR with fixed phi (only learn mu)
4. Compare performance

This tests whether fixing phi to the "correct" values (from Normal)
allows Tweedie to match Normal's performance.
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
    """Mean Absolute Scaled Error"""
    mae_forecast = np.mean(np.abs(forecast - actuals))
    naive_errors = np.abs(np.diff(training_series))
    mae_naive = np.mean(naive_errors)
    if mae_naive == 0 or np.isnan(mae_naive) or np.isinf(mae_naive):
        mae_naive = 1e-10
    mase = mae_forecast / mae_naive
    return mase

print("="*80)
print("EXPERIMENT: TWEEDIE WITH PHI FIXED TO NORMAL'S LEARNED PARAMETERS")
print("="*80)

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
print(f"  Scale: {scale}")
print(f"  Value range: [{ts.min():.2f}, {ts.max():.2f}]")

# Prepare datasets
train_data = [{"target": ts[:hist_length], "start": pd.Timestamp("2021-01-01")}]
test_data = [{"target": ts, "start": pd.Timestamp("2021-01-01")}]

train_ds = ListDataset(train_data, freq=freq)
test_ds = ListDataset(test_data, freq=freq)

# ============================================================================
# STEP 1: Train Normal DeepAR
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Train Normal DeepAR and Extract Learned Parameters")
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

# Generate forecast to extract learned parameters
print("\nGenerating Normal forecast and extracting learned parameters...")
forecast_normal = list(predictor_normal.predict(test_ds))[0]

# Extract mean and std from the forecast distribution
# The forecast object has samples; we can compute statistics from them
mean_normal = np.array(forecast_normal.mean)
# For Normal distribution, we can approximate std from quantiles
q10 = np.array(forecast_normal.quantile(0.1))
q90 = np.array(forecast_normal.quantile(0.9))
# For Normal: q90 - q10 ≈ 2.56 * std (80% interval)
std_normal = (q90 - q10) / 2.56

print(f"\nLearned Normal parameters (timestep-wise):")
print(f"  Mean range: [{mean_normal.min():.4f}, {mean_normal.max():.4f}]")
print(f"  Std range:  [{std_normal.min():.4f}, {std_normal.max():.4f}]")

# ============================================================================
# STEP 2: Convert to Tweedie parameters
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Convert Normal Parameters to Tweedie")
print("="*80)

# For Tweedie with p=1.5:
# Var[Y] = phi * mu^p
# Therefore: phi = var / mu^p = std^2 / mu^1.5

p = 1.5
var_normal = std_normal ** 2
phi_from_normal = var_normal / (mean_normal ** p)

# Use the median phi across timesteps as the fixed value
phi_fixed = np.median(phi_from_normal)

print(f"\nConverted Tweedie parameters:")
print(f"  phi (per timestep) range: [{phi_from_normal.min():.6f}, {phi_from_normal.max():.6f}]")
print(f"  phi (median across timesteps): {phi_fixed:.6f}")
print(f"  Using phi_fixed = {phi_fixed:.6f}")

# ============================================================================
# STEP 3: Train Tweedie with fixed phi
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Train Tweedie DeepAR with Fixed Phi (Only Learn Mu)")
print("="*80)
print(f"Using phi_fixed = {phi_fixed:.6f} (derived from Normal's learned parameters)")

# Reset random seed for fair comparison
np.random.seed(42)
torch.manual_seed(42)

estimator_tweedie_fixed = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    distr_output=TweedieOutput(p=1.5, phi_fixed=phi_fixed),
    trainer_kwargs={"max_epochs": 20},
)
predictor_tweedie_fixed = estimator_tweedie_fixed.train(train_ds)

# ============================================================================
# STEP 4: Compare all three approaches
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Compare Performance")
print("="*80)

# Also train Tweedie with learned phi for comparison
print("\nTraining Tweedie with learned phi for comparison...")
np.random.seed(42)
torch.manual_seed(42)

estimator_tweedie_learned = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    distr_output=TweedieOutput(p=1.5),  # No phi_fixed
    trainer_kwargs={"max_epochs": 20},
)
predictor_tweedie_learned = estimator_tweedie_learned.train(train_ds)

# Generate forecasts
print("\nGenerating forecasts...")
forecast_normal = list(predictor_normal.predict(test_ds))[0]
forecast_tweedie_fixed = list(predictor_tweedie_fixed.predict(test_ds))[0]
forecast_tweedie_learned = list(predictor_tweedie_learned.predict(test_ds))[0]

# Extract medians
median_normal = np.array(forecast_normal.median)
median_tweedie_fixed = np.array(forecast_tweedie_fixed.median)
median_tweedie_learned = np.array(forecast_tweedie_learned.median)

# Compute metrics
actuals = ts[hist_length:]
training = ts[:hist_length]

mase_normal = compute_mase(median_normal, actuals, training)
mase_tweedie_fixed = compute_mase(median_tweedie_fixed, actuals, training)
mase_tweedie_learned = compute_mase(median_tweedie_learned, actuals, training)

mae_normal = np.mean(np.abs(median_normal - actuals))
mae_tweedie_fixed = np.mean(np.abs(median_tweedie_fixed - actuals))
mae_tweedie_learned = np.mean(np.abs(median_tweedie_learned - actuals))

rmse_normal = np.sqrt(np.mean((median_normal - actuals)**2))
rmse_tweedie_fixed = np.sqrt(np.mean((median_tweedie_fixed - actuals)**2))
rmse_tweedie_learned = np.sqrt(np.mean((median_tweedie_learned - actuals)**2))

# Print results
print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nActual values range: [{actuals.min():.4f}, {actuals.max():.4f}]")
print(f"\n{'Approach':<30} {'MASE':<12} {'MAE':<12} {'RMSE':<12} {'Median Range'}")
print("-" * 90)
print(f"{'Normal (baseline)':<30} {mase_normal:<12.6f} {mae_normal:<12.6f} {rmse_normal:<12.6f} [{median_normal.min():.2f}, {median_normal.max():.2f}]")
print(f"{'Tweedie (learned phi)':<30} {mase_tweedie_learned:<12.6f} {mae_tweedie_learned:<12.6f} {rmse_tweedie_learned:<12.6f} [{median_tweedie_learned.min():.2f}, {median_tweedie_learned.max():.2f}]")
print(f"{'Tweedie (fixed phi=Normal)':<30} {mase_tweedie_fixed:<12.6f} {mae_tweedie_fixed:<12.6f} {rmse_tweedie_fixed:<12.6f} [{median_tweedie_fixed.min():.2f}, {median_tweedie_fixed.max():.2f}]")

# Analysis
print(f"\n{'='*80}")
print("ANALYSIS")
print('='*80)

pct_diff_learned = (mase_tweedie_learned - mase_normal) / mase_normal * 100
pct_diff_fixed = (mase_tweedie_fixed - mase_normal) / mase_normal * 100

print(f"\nMASE Comparison vs Normal:")
print(f"  Normal:                  {mase_normal:.6f} (baseline)")
print(f"  Tweedie (learned phi):   {mase_tweedie_learned:.6f} ({pct_diff_learned:+.1f}%)")
print(f"  Tweedie (fixed phi):     {mase_tweedie_fixed:.6f} ({pct_diff_fixed:+.1f}%)")

print(f"\nKey Question: Does fixing phi to Normal's parameters help?")
if abs(pct_diff_fixed) < abs(pct_diff_learned) and abs(pct_diff_fixed) < 20:
    print(f"  ✅ YES! Fixed phi ({pct_diff_fixed:+.1f}%) is much closer to Normal than learned phi ({pct_diff_learned:+.1f}%)")
    print(f"  → Fixing phi to correct values works!")
elif abs(pct_diff_fixed) < abs(pct_diff_learned):
    print(f"  ✓ PARTIAL. Fixed phi ({pct_diff_fixed:+.1f}%) is better than learned phi ({pct_diff_learned:+.1f}%), but not close to Normal")
    print(f"  → Fixed phi helps but doesn't fully solve the problem")
elif abs(pct_diff_fixed) < 20:
    print(f"  → Fixed phi ({pct_diff_fixed:+.1f}%) is close to Normal, comparable to learned phi ({pct_diff_learned:+.1f}%)")
else:
    print(f"  ❌ NO. Fixed phi ({pct_diff_fixed:+.1f}%) performs as badly or worse than learned phi ({pct_diff_learned:+.1f}%)")
    print(f"  → Issue may be with Tweedie implementation or phi-mu coupling")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Experiment: Tweedie with Phi Fixed to Normal Parameters', fontsize=14, fontweight='bold')

hist_x = np.arange(hist_length)
forecast_x = np.arange(hist_length, hist_length + prediction_length)

# Top left: Normal
ax = axes[0, 0]
ax.plot(hist_x, ts[:hist_length], 'o-', color='black', linewidth=1.5,
        markersize=2, label='Historical', alpha=0.7)
ax.plot(forecast_x, median_normal, color='purple', linewidth=2.5,
        label='Forecast', zorder=5)
ax.fill_between(forecast_x, forecast_normal.quantile(0.1),
                forecast_normal.quantile(0.9), alpha=0.3, color='purple')
ax.plot(forecast_x, actuals, 'o', color='red', markersize=4,
        label='Actual', alpha=0.8, zorder=6)
ax.axvline(x=hist_length, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('Time Step', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('Normal (Baseline)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
stats_text = f'MASE: {mase_normal:.4f}\nMAE: {mae_normal:.4f}'
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8), fontsize=9)

# Top right: Tweedie (learned phi)
ax = axes[0, 1]
ax.plot(hist_x, ts[:hist_length], 'o-', color='black', linewidth=1.5,
        markersize=2, label='Historical', alpha=0.7)
ax.plot(forecast_x, median_tweedie_learned, color='darkorange', linewidth=2.5,
        label='Forecast', zorder=5)
ax.fill_between(forecast_x, forecast_tweedie_learned.quantile(0.1),
                forecast_tweedie_learned.quantile(0.9), alpha=0.3, color='darkorange')
ax.plot(forecast_x, actuals, 'o', color='red', markersize=4,
        label='Actual', alpha=0.8, zorder=6)
ax.axvline(x=hist_length, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('Time Step', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('Tweedie (Learned phi)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
stats_text = f'MASE: {mase_tweedie_learned:.4f}\nMAE: {mae_tweedie_learned:.4f}\n{pct_diff_learned:+.1f}% vs Normal'
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.8), fontsize=9)

# Bottom left: Tweedie (fixed phi)
ax = axes[1, 0]
ax.plot(hist_x, ts[:hist_length], 'o-', color='black', linewidth=1.5,
        markersize=2, label='Historical', alpha=0.7)
ax.plot(forecast_x, median_tweedie_fixed, color='green', linewidth=2.5,
        label='Forecast', zorder=5)
ax.fill_between(forecast_x, forecast_tweedie_fixed.quantile(0.1),
                forecast_tweedie_fixed.quantile(0.9), alpha=0.3, color='green')
ax.plot(forecast_x, actuals, 'o', color='red', markersize=4,
        label='Actual', alpha=0.8, zorder=6)
ax.axvline(x=hist_length, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('Time Step', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title(f'Tweedie (Fixed phi={phi_fixed:.4f})', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
stats_text = f'MASE: {mase_tweedie_fixed:.4f}\nMAE: {mae_tweedie_fixed:.4f}\n{pct_diff_fixed:+.1f}% vs Normal'
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=9)

# Bottom right: Direct comparison
ax = axes[1, 1]
ax.plot(forecast_x, actuals, 'o-', color='red', linewidth=2.5, markersize=5,
        label='Actual', zorder=10, alpha=0.8)
ax.plot(forecast_x, median_normal, 's-', color='purple', linewidth=2,
        markersize=4, label='Normal', alpha=0.7)
ax.plot(forecast_x, median_tweedie_learned, '^-', color='darkorange', linewidth=2,
        markersize=4, label='Tweedie (learned)', alpha=0.7)
ax.plot(forecast_x, median_tweedie_fixed, 'D-', color='green', linewidth=2,
        markersize=4, label='Tweedie (fixed)', alpha=0.7)
ax.set_xlabel('Time Step', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('Direct Comparison', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/TSForecasting/tweedie_fixed_to_normal_params.png',
            dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved: tweedie_fixed_to_normal_params.png")

# Conclusion
print(f"\n{'='*80}")
print("CONCLUSION")
print('='*80)

if abs(pct_diff_fixed) < 10:
    print(f"✅ SUCCESS: Fixing phi to Normal's learned parameters works!")
    print(f"   Tweedie (fixed phi) achieves {pct_diff_fixed:+.1f}% vs Normal (within 10%)")
    print(f"   This proves the issue is the phi gradient bug, not the Tweedie distribution itself.")
    print(f"\n   Implication: Option 1 CAN work if we estimate phi correctly!")
elif abs(pct_diff_fixed) < abs(pct_diff_learned):
    print(f"✓ IMPROVEMENT: Fixed phi ({pct_diff_fixed:+.1f}%) is better than learned phi ({pct_diff_learned:+.1f}%)")
    print(f"   But still not close enough to Normal to be practically useful.")
    print(f"   May need better phi estimation or different approach.")
else:
    print(f"❌ NO IMPROVEMENT: Fixed phi ({pct_diff_fixed:+.1f}%) doesn't help vs learned phi ({pct_diff_learned:+.1f}%)")
    print(f"   Issue may be deeper than just the gradient bug.")
    print(f"   Could be phi-mu coupling in scaled space or Tweedie implementation issue.")

print(f"\nFinal recommendation:")
if abs(pct_diff_fixed) < 20:
    print(f"  → Fixed phi approach shows promise! Investigate better phi estimation methods.")
else:
    print(f"  → Use NegativeBinomial distribution instead of Tweedie.")
