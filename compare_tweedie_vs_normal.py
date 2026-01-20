#!/usr/bin/env python
"""
Compare Tweedie (learned phi) vs Normal (default DeepAR) on clean sinusoid.

This tests whether the Tweedie distribution with learned phi can match
the performance of the default Normal distribution, despite the known
phi gradient bug.
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
print("TWEEDIE (LEARNED PHI) VS NORMAL ON CLEAN SINUSOID")
print("="*80)

# Set random seeds for reproducibility
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
print(f"  Mean: {ts[:hist_length].mean():.2f}")
print(f"  Std: {ts[:hist_length].std():.2f}")

# Prepare datasets
train_data_normal = [{"target": ts[:hist_length], "start": pd.Timestamp("2021-01-01")}]
train_data_tweedie = [{"target": ts[:hist_length], "start": pd.Timestamp("2021-01-01")}]
test_data = [{"target": ts, "start": pd.Timestamp("2021-01-01")}]

train_ds_normal = ListDataset(train_data_normal, freq=freq)
train_ds_tweedie = ListDataset(train_data_tweedie, freq=freq)
test_ds = ListDataset(test_data, freq=freq)

# Train Normal model (default DeepAR)
print("\n" + "="*80)
print("1. Training Default DeepAR with Normal Distribution")
print("="*80)
estimator_normal = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    distr_output=NormalOutput(),
    trainer_kwargs={"max_epochs": 20},
)
predictor_normal = estimator_normal.train(train_ds_normal)

# Train Tweedie model (learned phi)
print("\n" + "="*80)
print("2. Training DeepAR with Tweedie Distribution (p=1.5, learned phi)")
print("="*80)
print("Note: phi will be learned via gradient descent (despite gradient bug)")
estimator_tweedie = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    distr_output=TweedieOutput(p=1.5),  # No phi_fixed - will learn phi
    trainer_kwargs={"max_epochs": 20},
)
predictor_tweedie = estimator_tweedie.train(train_ds_tweedie)

# Generate forecasts
print("\nGenerating forecasts...")
forecast_normal = list(predictor_normal.predict(test_ds))[0]
forecast_tweedie = list(predictor_tweedie.predict(test_ds))[0]

# Extract median forecasts and quantiles
median_normal = np.array(forecast_normal.median)
median_tweedie = np.array(forecast_tweedie.median)

q10_normal = np.array(forecast_normal.quantile(0.1))
q90_normal = np.array(forecast_normal.quantile(0.9))
q10_tweedie = np.array(forecast_tweedie.quantile(0.1))
q90_tweedie = np.array(forecast_tweedie.quantile(0.9))

# Compute metrics
actuals = ts[hist_length:]
training = ts[:hist_length]

mase_normal = compute_mase(median_normal, actuals, training)
mase_tweedie = compute_mase(median_tweedie, actuals, training)

mae_normal = np.mean(np.abs(median_normal - actuals))
mae_tweedie = np.mean(np.abs(median_tweedie - actuals))

rmse_normal = np.sqrt(np.mean((median_normal - actuals)**2))
rmse_tweedie = np.sqrt(np.mean((median_tweedie - actuals)**2))

# Compute coverage (% of actuals within 80% PI)
in_pi_normal = np.sum((actuals >= q10_normal) & (actuals <= q90_normal)) / len(actuals) * 100
in_pi_tweedie = np.sum((actuals >= q10_tweedie) & (actuals <= q90_tweedie)) / len(actuals) * 100

# Print results
print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nActual values range: [{actuals.min():.4f}, {actuals.max():.4f}]")
print(f"\n{'Metric':<20} {'Normal':<15} {'Tweedie':<15} {'Difference'}")
print("-" * 70)
print(f"{'MASE':<20} {mase_normal:<15.6f} {mase_tweedie:<15.6f} {abs(mase_tweedie - mase_normal):.6f}")
print(f"{'MAE':<20} {mae_normal:<15.6f} {mae_tweedie:<15.6f} {abs(mae_tweedie - mae_normal):.6f}")
print(f"{'RMSE':<20} {rmse_normal:<15.6f} {rmse_tweedie:<15.6f} {abs(rmse_tweedie - rmse_normal):.6f}")
print(f"{'80% PI Coverage':<20} {in_pi_normal:<15.1f}% {in_pi_tweedie:<15.1f}% {abs(in_pi_tweedie - in_pi_normal):.1f}%")

print(f"\n{'Forecast Range':<20} {'Normal':<30} {'Tweedie':<30}")
print("-" * 80)
print(f"{'Median':<20} [{median_normal.min():.4f}, {median_normal.max():.4f}]   [{median_tweedie.min():.4f}, {median_tweedie.max():.4f}]")
print(f"{'10th percentile':<20} [{q10_normal.min():.4f}, {q10_normal.max():.4f}]   [{q10_tweedie.min():.4f}, {q10_tweedie.max():.4f}]")
print(f"{'90th percentile':<20} [{q90_normal.min():.4f}, {q90_normal.max():.4f}]   [{q90_tweedie.min():.4f}, {q90_tweedie.max():.4f}]")

# Analysis
print(f"\n{'='*80}")
print("ANALYSIS")
print('='*80)

# MASE comparison
pct_diff = (mase_tweedie - mase_normal) / mase_normal * 100
if abs(pct_diff) < 10:
    verdict = "✅ EXCELLENT - Nearly identical performance"
elif abs(pct_diff) < 30:
    verdict = "✓ GOOD - Similar performance despite gradient bug"
elif abs(pct_diff) < 100:
    verdict = "⚠️ MODERATE - Noticeable difference"
else:
    verdict = "❌ POOR - Significant performance gap"

print(f"\nMASE Comparison:")
print(f"  Normal:  {mase_normal:.6f}")
print(f"  Tweedie: {mase_tweedie:.6f}")
print(f"  Difference: {pct_diff:+.1f}%")
print(f"  {verdict}")

# Coverage comparison
print(f"\n80% Prediction Interval Coverage:")
print(f"  Normal:  {in_pi_normal:.1f}% (target: 80%)")
print(f"  Tweedie: {in_pi_tweedie:.1f}% (target: 80%)")
if abs(in_pi_normal - 80) < 10 and abs(in_pi_tweedie - 80) < 10:
    print(f"  ✓ Both distributions provide well-calibrated uncertainty estimates")
elif abs(in_pi_tweedie - 80) > abs(in_pi_normal - 80) + 20:
    print(f"  ⚠️ Tweedie uncertainty estimates are less calibrated")
else:
    print(f"  → Similar calibration quality")

# Point forecast accuracy
print(f"\nPoint Forecast Accuracy:")
if mase_tweedie < mase_normal:
    improvement = (mase_normal - mase_tweedie) / mase_normal * 100
    print(f"  🎉 Tweedie is {improvement:.1f}% better than Normal!")
elif mase_normal < mase_tweedie:
    degradation = (mase_tweedie - mase_normal) / mase_normal * 100
    if degradation < 30:
        print(f"  → Tweedie is {degradation:.1f}% worse (acceptable for practical use)")
    else:
        print(f"  ⚠️ Tweedie is {degradation:.1f}% worse (may be problematic)")
else:
    print(f"  → Identical performance")

# Create detailed visualization
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

hist_x = np.arange(hist_length)
forecast_x = np.arange(hist_length, hist_length + prediction_length)

# Top row: Full forecasts
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(hist_x, ts[:hist_length], 'o-', color='black', linewidth=1.5,
         markersize=2, label='Historical', alpha=0.7)
ax1.plot(forecast_x, median_normal, color='purple', linewidth=2.5,
         label='Forecast (Median)', zorder=5)
ax1.fill_between(forecast_x, q10_normal, q90_normal, alpha=0.3,
                 color='purple', label='80% PI')
ax1.plot(forecast_x, actuals, 'o', color='red', markersize=4,
         label='Actual', alpha=0.8, zorder=6)
ax1.axvline(x=hist_length, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax1.set_xlabel('Time Step', fontsize=11)
ax1.set_ylabel('Value', fontsize=11)
ax1.set_title('Normal Distribution (Default DeepAR)', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
stats_text = f'MASE: {mase_normal:.4f}\nMAE: {mae_normal:.4f}\nRMSE: {rmse_normal:.4f}\nCoverage: {in_pi_normal:.1f}%'
ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8), fontsize=9)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(hist_x, ts[:hist_length], 'o-', color='black', linewidth=1.5,
         markersize=2, label='Historical', alpha=0.7)
ax2.plot(forecast_x, median_tweedie, color='darkorange', linewidth=2.5,
         label='Forecast (Median)', zorder=5)
ax2.fill_between(forecast_x, q10_tweedie, q90_tweedie, alpha=0.3,
                 color='darkorange', label='80% PI')
ax2.plot(forecast_x, actuals, 'o', color='red', markersize=4,
         label='Actual', alpha=0.8, zorder=6)
ax2.axvline(x=hist_length, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_xlabel('Time Step', fontsize=11)
ax2.set_ylabel('Value', fontsize=11)
ax2.set_title('Tweedie Distribution (p=1.5, Learned phi)', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
stats_text = f'MASE: {mase_tweedie:.4f}\nMAE: {mae_tweedie:.4f}\nRMSE: {rmse_tweedie:.4f}\nCoverage: {in_pi_tweedie:.1f}%'
ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.8), fontsize=9)

# Middle row: Forecast errors
ax3 = fig.add_subplot(gs[1, 0])
errors_normal = median_normal - actuals
errors_tweedie = median_tweedie - actuals
ax3.plot(forecast_x, errors_normal, 'o-', color='purple', linewidth=2,
         markersize=4, label='Normal', alpha=0.7)
ax3.plot(forecast_x, errors_tweedie, 's-', color='darkorange', linewidth=2,
         markersize=4, label='Tweedie', alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_xlabel('Time Step', fontsize=11)
ax3.set_ylabel('Forecast Error', fontsize=11)
ax3.set_title('Forecast Errors (Median - Actual)', fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3)

# Middle row: Error distribution
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(errors_normal, bins=15, alpha=0.6, color='purple', label='Normal', edgecolor='black')
ax4.hist(errors_tweedie, bins=15, alpha=0.6, color='darkorange', label='Tweedie', edgecolor='black')
ax4.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax4.axvline(x=errors_normal.mean(), color='purple', linestyle='-', linewidth=2,
            label=f'Normal mean: {errors_normal.mean():.3f}')
ax4.axvline(x=errors_tweedie.mean(), color='darkorange', linestyle='-', linewidth=2,
            label=f'Tweedie mean: {errors_tweedie.mean():.3f}')
ax4.set_xlabel('Forecast Error', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax4.legend(loc='best', fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# Bottom row: Direct comparison
ax5 = fig.add_subplot(gs[2, :])
ax5.plot(forecast_x, actuals, 'o-', color='red', linewidth=2.5, markersize=5,
         label='Actual', zorder=10, alpha=0.8)
ax5.plot(forecast_x, median_normal, 's-', color='purple', linewidth=2,
         markersize=4, label='Normal Forecast', alpha=0.7)
ax5.plot(forecast_x, median_tweedie, '^-', color='darkorange', linewidth=2,
         markersize=4, label='Tweedie Forecast', alpha=0.7)
ax5.set_xlabel('Time Step', fontsize=11)
ax5.set_ylabel('Value', fontsize=11)
ax5.set_title('Direct Comparison: Actual vs Forecasts', fontsize=12, fontweight='bold')
ax5.legend(loc='best', fontsize=10)
ax5.grid(True, alpha=0.3)

# Add overall title with key result
fig.suptitle(f'Tweedie (Learned phi) vs Normal on Clean Sinusoid | MASE Difference: {pct_diff:+.1f}%',
             fontsize=14, fontweight='bold', y=0.995)

plt.savefig('/home/user/TSForecasting/tweedie_vs_normal_comparison.png',
            dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved: tweedie_vs_normal_comparison.png")

# Summary
print(f"\n{'='*80}")
print("CONCLUSION")
print('='*80)

if abs(pct_diff) < 30:
    print(f"✅ Tweedie (learned phi) performs acceptably despite the gradient bug!")
    print(f"   The {abs(pct_diff):.1f}% difference is small enough for practical use.")
    print(f"   The mathematical error in phi gradients doesn't prevent decent forecasting.")
else:
    print(f"⚠️ Tweedie (learned phi) shows significant degradation ({abs(pct_diff):.1f}%).")
    print(f"   The gradient bug may be causing performance issues.")
    print(f"   Consider using NegativeBinomial instead.")

print(f"\nRecommendation:")
if abs(pct_diff) < 30:
    print(f"  → Tweedie can be used for your forecasting task")
    print(f"  → Document the gradient bug but don't block on fixing it")
    print(f"  → Test on real-world noisy data to confirm")
else:
    print(f"  → Switch to NegativeBinomial distribution")
    print(f"  → Don't invest time fixing the Tweedie gradient bug")
