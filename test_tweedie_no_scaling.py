#!/usr/bin/env python
"""
Test TweedieFull WITHOUT any normalization/scaling.

Hypothesis: The series expansion works correctly (parameter recovery proves it),
but fails when combined with DeepAR's MeanScaler. Let's remove scaling entirely
and see if we get near-perfect forecasts.

Setup:
- Single clean sinusoid (no noise)
- NO MeanScaler (set to None)
- Compare: Normal vs TweedieFull
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
print("TEST: TweedieFull WITHOUT Normalization")
print("="*80)
print("\nHypothesis: Series expansion is correct (parameter recovery proves it),")
print("but fails with MeanScaler. Let's test without any scaling.\n")

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Generate clean sinusoidal series
freq = "H"
prediction_length = 24
hist_length = 150
n_timesteps = hist_length + prediction_length

print("Generating perfectly clean sinusoidal series...")
t = np.arange(n_timesteps)
scale = 10
ts = scale + (scale * 0.3) * np.sin(2 * np.pi * t / 24)

print(f"  Training length: {hist_length}")
print(f"  Prediction length: {prediction_length}")
print(f"  Value range: [{ts.min():.2f}, {ts.max():.2f}]")
print(f"  Mean: {ts[:hist_length].mean():.2f}")
print(f"  Std: {ts[:hist_length].std():.2f}")

# Prepare datasets
train_data = [{"target": ts[:hist_length], "start": pd.Timestamp("2021-01-01")}]
test_data = [{"target": ts, "start": pd.Timestamp("2021-01-01")}]

train_ds = ListDataset(train_data, freq=freq)
test_ds = ListDataset(test_data, freq=freq)

# Train Normal model WITHOUT scaling
print("\n" + "="*80)
print("1. Training DeepAR with Normal (NO SCALING)")
print("="*80)
print("Setting scaling=False to disable MeanScaler...")

estimator_normal = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    distr_output=NormalOutput(),
    scaling=False,  # ← DISABLE SCALING
    trainer_kwargs={"max_epochs": 20},
)
predictor_normal = estimator_normal.train(train_ds)

# Train TweedieFull WITHOUT scaling
print("\n" + "="*80)
print("2. Training DeepAR with TweedieFull (NO SCALING)")
print("="*80)
print("Using series expansion with correct gradients, no MeanScaler...")
np.random.seed(42)
torch.manual_seed(42)

estimator_tweedie = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    distr_output=TweedieFullOutput(p=1.5),
    scaling=False,  # ← DISABLE SCALING
    trainer_kwargs={"max_epochs": 20},
)
predictor_tweedie = estimator_tweedie.train(train_ds)

# Generate forecasts
print("\nGenerating forecasts...")
forecast_normal = list(predictor_normal.predict(test_ds))[0]
forecast_tweedie = list(predictor_tweedie.predict(test_ds))[0]

# Extract medians
median_normal = np.array(forecast_normal.median)
median_tweedie = np.array(forecast_tweedie.median)

# Compute metrics
actuals = ts[hist_length:]
training = ts[:hist_length]

mase_normal = compute_mase(median_normal, actuals, training)
mase_tweedie = compute_mase(median_tweedie, actuals, training)

mae_normal = np.mean(np.abs(median_normal - actuals))
mae_tweedie = np.mean(np.abs(median_tweedie - actuals))

rmse_normal = np.sqrt(np.mean((median_normal - actuals)**2))
rmse_tweedie = np.sqrt(np.mean((median_tweedie - actuals)**2))

# Compute prediction interval coverage
q10_normal = np.array(forecast_normal.quantile(0.1))
q90_normal = np.array(forecast_normal.quantile(0.9))
q10_tweedie = np.array(forecast_tweedie.quantile(0.1))
q90_tweedie = np.array(forecast_tweedie.quantile(0.9))

coverage_normal = np.sum((actuals >= q10_normal) & (actuals <= q90_normal)) / len(actuals) * 100
coverage_tweedie = np.sum((actuals >= q10_tweedie) & (actuals <= q90_tweedie)) / len(actuals) * 100

# Print results
print("\n" + "="*80)
print("RESULTS (NO SCALING)")
print("="*80)

print(f"\nActual values range: [{actuals.min():.4f}, {actuals.max():.4f}]")
print(f"\n{'Distribution':<20} {'MASE':<12} {'MAE':<12} {'RMSE':<12} {'80% Coverage'}")
print("-" * 75)
print(f"{'Normal':<20} {mase_normal:<12.6f} {mae_normal:<12.6f} {rmse_normal:<12.6f} {coverage_normal:>6.1f}%")
print(f"{'TweedieFull':<20} {mase_tweedie:<12.6f} {mae_tweedie:<12.6f} {rmse_tweedie:<12.6f} {coverage_tweedie:>6.1f}%")

# Analysis
print(f"\n{'='*80}")
print("ANALYSIS")
print('='*80)

pct_diff = (mase_tweedie - mase_normal) / mase_normal * 100

print(f"\nMASE Comparison:")
print(f"  Normal:      {mase_normal:.6f}")
print(f"  TweedieFull: {mase_tweedie:.6f} ({pct_diff:+.1f}%)")

print(f"\nForecast Ranges:")
print(f"  Normal:      [{median_normal.min():.2f}, {median_normal.max():.2f}]")
print(f"  TweedieFull: [{median_tweedie.min():.2f}, {median_tweedie.max():.2f}]")
print(f"  Actual:      [{actuals.min():.2f}, {actuals.max():.2f}]")

print(f"\nKey Question: Does removing scaling fix TweedieFull?")
if abs(pct_diff) < 10:
    print(f"  ✅ YES! TweedieFull ({pct_diff:+.1f}%) matches Normal (within 10%)")
    print(f"  → The issue WAS the MeanScaler interaction!")
    print(f"  → Series expansion works correctly without scaling")
elif abs(pct_diff) < 30:
    print(f"  ✓ BETTER! TweedieFull ({pct_diff:+.1f}%) is close to Normal (within 30%)")
    print(f"  → Scaling was a major issue, but may be other factors")
elif mase_tweedie < 10:
    print(f"  → TweedieFull works reasonably ({pct_diff:+.1f}%), but not as good as Normal")
    print(f"  → Scaling helped but other issues remain")
else:
    print(f"  ❌ NO. TweedieFull still fails (MASE {mase_tweedie:.2f})")
    print(f"  → Issue is not just scaling")

# Compare with previous tests
print(f"\n{'='*80}")
print("COMPARISON WITH PREVIOUS TESTS")
print('='*80)

print(f"\nTweedieFull performance across tests:")
print(f"  Parameter recovery:     ✅ mu=5.4% error, phi=7.1% error")
print(f"  DeepAR WITH scaling:    ❌ MASE 151.87 (+6802%)")
print(f"  DeepAR WITHOUT scaling: {'✅' if abs(pct_diff) < 30 else '❌'} MASE {mase_tweedie:.2f} ({pct_diff:+.1f}%)")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f'TweedieFull WITHOUT Scaling Test | Difference: {pct_diff:+.1f}%',
             fontsize=14, fontweight='bold')

hist_x = np.arange(hist_length)
forecast_x = np.arange(hist_length, hist_length + prediction_length)

# Top left: Normal
ax = axes[0, 0]
ax.plot(hist_x, ts[:hist_length], 'o-', color='black', linewidth=1.5,
        markersize=2, label='Historical', alpha=0.7)
ax.plot(forecast_x, median_normal, color='purple', linewidth=2.5,
        label='Forecast', zorder=5)
ax.fill_between(forecast_x, q10_normal, q90_normal, alpha=0.3, color='purple')
ax.plot(forecast_x, actuals, 'o', color='red', markersize=4,
        label='Actual', alpha=0.8, zorder=6)
ax.axvline(x=hist_length, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('Time Step', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('Normal (NO SCALING)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
stats_text = f'MASE: {mase_normal:.4f}\nMAE: {mae_normal:.4f}\nCoverage: {coverage_normal:.1f}%'
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8), fontsize=9)

# Top right: TweedieFull
ax = axes[0, 1]
ax.plot(hist_x, ts[:hist_length], 'o-', color='black', linewidth=1.5,
        markersize=2, label='Historical', alpha=0.7)
ax.plot(forecast_x, median_tweedie, color='green', linewidth=2.5,
        label='Forecast', zorder=5)
ax.fill_between(forecast_x, q10_tweedie, q90_tweedie, alpha=0.3, color='green')
ax.plot(forecast_x, actuals, 'o', color='red', markersize=4,
        label='Actual', alpha=0.8, zorder=6)
ax.axvline(x=hist_length, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('Time Step', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('TweedieFull (NO SCALING)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
stats_text = f'MASE: {mase_tweedie:.4f} ({pct_diff:+.1f}%)\nMAE: {mae_tweedie:.4f}\nCoverage: {coverage_tweedie:.1f}%'
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=9)

# Bottom left: Errors
ax = axes[1, 0]
errors_normal = median_normal - actuals
errors_tweedie = median_tweedie - actuals
ax.plot(forecast_x, errors_normal, 'o-', color='purple', linewidth=2,
        markersize=4, label='Normal', alpha=0.7)
ax.plot(forecast_x, errors_tweedie, 's-', color='green', linewidth=2,
        markersize=4, label='TweedieFull', alpha=0.7)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Time Step', fontsize=11)
ax.set_ylabel('Forecast Error', fontsize=11)
ax.set_title('Forecast Errors (Median - Actual)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

# Bottom right: Direct comparison
ax = axes[1, 1]
ax.plot(forecast_x, actuals, 'o-', color='red', linewidth=2.5, markersize=5,
        label='Actual', zorder=10, alpha=0.8)
ax.plot(forecast_x, median_normal, 's-', color='purple', linewidth=2,
        markersize=4, label='Normal', alpha=0.7)
ax.plot(forecast_x, median_tweedie, 'D-', color='green', linewidth=2,
        markersize=4, label='TweedieFull', alpha=0.7)
ax.set_xlabel('Time Step', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('Direct Comparison', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/TSForecasting/tweedie_no_scaling_test.png',
            dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved: tweedie_no_scaling_test.png")

# Conclusion
print(f"\n{'='*80}")
print("CONCLUSION")
print('='*80)

if abs(pct_diff) < 10:
    print(f"\n🎉 BREAKTHROUGH! Removing scaling fixes TweedieFull!")
    print(f"   TweedieFull achieves {pct_diff:+.1f}% vs Normal (within 10%)")
    print(f"\n   Root cause identified:")
    print(f"   - Series expansion is mathematically correct")
    print(f"   - Parameter recovery works (mu=5.4%, phi=7.1% error)")
    print(f"   - Issue was interaction with MeanScaler")
    print(f"\n   Next steps:")
    print(f"   - Fix series expansion to work correctly with scaled data")
    print(f"   - Or use TweedieFull without scaling (limited use case)")
elif abs(pct_diff) < 30:
    print(f"\n✓ MAJOR IMPROVEMENT! Scaling was a significant issue.")
    print(f"   TweedieFull: {pct_diff:+.1f}% vs Normal")
    print(f"   (Was +6802% with scaling, now {pct_diff:+.1f}% without)")
    print(f"\n   Scaling was the main problem, but refinement needed.")
else:
    print(f"\n⚠️ Scaling wasn't the only issue.")
    print(f"   TweedieFull: {pct_diff:+.1f}% vs Normal")
    print(f"   Other factors affecting performance:")
    print(f"   - Series bounds calculation")
    print(f"   - Numerical stability")
    print(f"   - Initialization")
