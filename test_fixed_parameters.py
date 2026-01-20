#!/usr/bin/env python
"""
Test if Tweedie can reproduce Normal results when parameters are fixed.
This diagnoses whether the issue is in the Tweedie distribution itself or in how DeepAR learns it.
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

# Generate the same clean sinusoidal series
freq = "H"
prediction_length = 24
hist_length = 150
n_timesteps = hist_length + prediction_length

t = np.arange(n_timesteps)
scale = 10
ts = scale + (scale * 0.3) * np.sin(2 * np.pi * t / 24)

print("\nGenerating perfectly clean sinusoidal series...")
print(f"  Value range: [{ts.min():.2f}, {ts.max():.2f}]")

# Prepare datasets
train_data = [{
    "target": ts[:hist_length],
    "start": pd.Timestamp("2021-01-01")
}]
test_data = [{
    "target": ts,
    "start": pd.Timestamp("2021-01-01")
}]

train_ds = ListDataset(train_data, freq=freq)
test_ds = ListDataset(test_data, freq=freq)

# Train Normal model
print("\n" + "="*70)
print("STEP 1: Train Normal Distribution to get reference parameters")
print("="*70)
estimator_normal = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    distr_output=NormalOutput(),
    trainer_kwargs={"max_epochs": 20},
)
predictor_normal = estimator_normal.train(train_ds)

# Get Normal distribution's forecast
print("\nGenerating Normal forecast...")
forecast_normal = list(predictor_normal.predict(test_ds))[0]

# Extract the distribution parameters from Normal
print("\n" + "="*70)
print("STEP 2: Extract Normal's learned parameters")
print("="*70)

# The forecast object contains samples - we can analyze the distribution
mean_normal = np.array(forecast_normal.mean)
median_normal = np.array(forecast_normal.median)

# Get quantiles to estimate standard deviation
q10 = np.array(forecast_normal.quantile(0.1))
q90 = np.array(forecast_normal.quantile(0.9))
# For normal distribution: P(X < q90) = 0.9 => q90 = mean + 1.28*sigma
# So: sigma ≈ (q90 - q10) / 2.56
std_normal = (q90 - q10) / 2.56

print(f"\nNormal Distribution Parameters (averaged over forecast horizon):")
print(f"  Mean: {mean_normal.mean():.4f} ± {mean_normal.std():.4f}")
print(f"  Std: {std_normal.mean():.4f} ± {std_normal.std():.4f}")
print(f"  Median: {median_normal.mean():.4f} ± {median_normal.std():.4f}")

# Compute MASE for Normal
actuals = ts[hist_length:]
training = ts[:hist_length]
mase_normal = compute_mase(median_normal, actuals, training)
print(f"  MASE: {mase_normal:.6f}")

# Now train Tweedie normally (for comparison)
print("\n" + "="*70)
print("STEP 3: Train Tweedie Distribution (baseline)")
print("="*70)
estimator_tweedie_learned = DeepAREstimator(
    freq=freq,
    prediction_length=prediction_length,
    num_layers=2,
    hidden_size=40,
    distr_output=TweedieOutput(p=1.5),
    trainer_kwargs={"max_epochs": 20},
)
predictor_tweedie_learned = estimator_tweedie_learned.train(train_ds)

print("\nGenerating Tweedie forecast (learned parameters)...")
forecast_tweedie_learned = list(predictor_tweedie_learned.predict(test_ds))[0]
median_tweedie_learned = np.array(forecast_tweedie_learned.median)
mase_tweedie_learned = compute_mase(median_tweedie_learned, actuals, training)

# Extract Tweedie's learned parameters
mean_tweedie_learned = np.array(forecast_tweedie_learned.mean)
print(f"\nTweedie (Learned) Parameters:")
print(f"  Mean: {mean_tweedie_learned.mean():.4f} ± {mean_tweedie_learned.std():.4f}")
print(f"  Median: {median_tweedie_learned.mean():.4f} ± {median_tweedie_learned.std():.4f}")
print(f"  MASE: {mase_tweedie_learned:.6f}")

# Now create a custom test: manually set Tweedie parameters to match Normal
print("\n" + "="*70)
print("STEP 4: Create Tweedie samples with FIXED parameters matching Normal")
print("="*70)

# Import Tweedie distribution
from gluonts.torch.distributions.tweedie import Tweedie

# Create Tweedie distribution with fixed parameters
# We'll use the mean from Normal, and estimate scale parameter
print("\nCreating Tweedie distribution with fixed parameters...")
print(f"  Setting mu (mean) = {mean_normal.mean():.4f}")

# For Tweedie with p=1.5:
# Var(Y) = φ * μ^p = φ * μ^1.5
# We want Var(Y) ≈ std_normal^2
# So: φ = std_normal^2 / μ^1.5
target_variance = std_normal.mean()**2
target_mean = mean_normal.mean()
phi = target_variance / (target_mean**1.5)
print(f"  Setting φ (scale) = {phi:.6f} (based on Normal's variance)")

# Create fixed parameter Tweedie distributions for each timestep
fixed_tweedie_samples = []
num_samples = 100

for t_idx in range(prediction_length):
    mu_t = mean_normal[t_idx]
    var_t = std_normal[t_idx]**2
    phi_t = var_t / (mu_t**1.5) if mu_t > 0 else 1.0

    # Create Tweedie distribution
    tweedie_dist = Tweedie(
        mu=torch.tensor([mu_t]),
        p=torch.tensor([1.5]),
        phi=torch.tensor([phi_t])
    )

    # Sample from it
    samples_t = tweedie_dist.sample((num_samples,)).numpy().flatten()
    fixed_tweedie_samples.append(samples_t)

fixed_tweedie_samples = np.array(fixed_tweedie_samples).T  # Shape: (num_samples, prediction_length)

# Compute median and quantiles from fixed samples
median_tweedie_fixed = np.median(fixed_tweedie_samples, axis=0)
mean_tweedie_fixed = np.mean(fixed_tweedie_samples, axis=0)
q10_tweedie_fixed = np.percentile(fixed_tweedie_samples, 10, axis=0)
q90_tweedie_fixed = np.percentile(fixed_tweedie_samples, 90, axis=0)
q25_tweedie_fixed = np.percentile(fixed_tweedie_samples, 25, axis=0)
q75_tweedie_fixed = np.percentile(fixed_tweedie_samples, 75, axis=0)

mase_tweedie_fixed = compute_mase(median_tweedie_fixed, actuals, training)

print(f"\nTweedie (Fixed Parameters) Results:")
print(f"  Mean: {mean_tweedie_fixed.mean():.4f} ± {mean_tweedie_fixed.std():.4f}")
print(f"  Median: {median_tweedie_fixed.mean():.4f} ± {median_tweedie_fixed.std():.4f}")
print(f"  MASE: {mase_tweedie_fixed:.6f}")

# Print comparison
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"{'Distribution':<30} {'MASE':<12} {'Mean':<12} {'Median':<12}")
print("-" * 70)
print(f"{'Normal (trained)':<30} {mase_normal:<12.6f} {mean_normal.mean():<12.4f} {median_normal.mean():<12.4f}")
print(f"{'Tweedie (trained params)':<30} {mase_tweedie_learned:<12.6f} {mean_tweedie_learned.mean():<12.4f} {median_tweedie_learned.mean():<12.4f}")
print(f"{'Tweedie (fixed to Normal)':<30} {mase_tweedie_fixed:<12.6f} {mean_tweedie_fixed.mean():<12.4f} {median_tweedie_fixed.mean():<12.4f}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
if abs(mase_tweedie_fixed - mase_normal) < 0.5:
    print("✓ SUCCESS: Tweedie with fixed parameters achieves similar MASE to Normal!")
    print("  This suggests the Tweedie distribution itself is fine.")
    print("  The problem is likely in how DeepAR learns Tweedie parameters.")
else:
    print("✗ ISSUE: Even with fixed parameters, Tweedie differs from Normal.")
    print("  This suggests a potential issue with the Tweedie distribution itself.")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Diagnostic Test: Fixed vs Learned Parameters',
             fontsize=14, fontweight='bold')

hist_x = np.arange(hist_length)
forecast_x = np.arange(hist_length, hist_length + prediction_length)

# Plot 1: Normal (trained)
ax1 = axes[0]
ax1.plot(hist_x, ts[:hist_length], 'o-', color='black',
         linewidth=1.5, markersize=3, label='Historical', alpha=0.8)
ax1.plot(forecast_x, forecast_normal.median, color='purple',
         linewidth=2.5, label='Forecast (Median)', zorder=5)
ax1.fill_between(forecast_x, forecast_normal.quantile(0.1),
                 forecast_normal.quantile(0.9), alpha=0.25,
                 color='purple', label='80% PI')
ax1.fill_between(forecast_x, forecast_normal.quantile(0.25),
                 forecast_normal.quantile(0.75), alpha=0.4,
                 color='purple', label='50% PI')
ax1.plot(forecast_x, actuals, 'o', color='red',
         markersize=5, label='Actual', alpha=0.7, zorder=6)
ax1.axvline(x=hist_length, color='green', linestyle='--',
            linewidth=2, alpha=0.7)
ax1.set_xlabel('Time Step', fontsize=11)
ax1.set_ylabel('Value', fontsize=11)
ax1.set_title(f'Normal (Trained)\nMASE: {mase_normal:.4f}',
              fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Tweedie (trained)
ax2 = axes[1]
ax2.plot(hist_x, ts[:hist_length], 'o-', color='black',
         linewidth=1.5, markersize=3, label='Historical', alpha=0.8)
ax2.plot(forecast_x, forecast_tweedie_learned.median, color='blue',
         linewidth=2.5, label='Forecast (Median)', zorder=5)
ax2.fill_between(forecast_x, forecast_tweedie_learned.quantile(0.1),
                 forecast_tweedie_learned.quantile(0.9), alpha=0.25,
                 color='blue', label='80% PI')
ax2.fill_between(forecast_x, forecast_tweedie_learned.quantile(0.25),
                 forecast_tweedie_learned.quantile(0.75), alpha=0.4,
                 color='blue', label='50% PI')
ax2.plot(forecast_x, actuals, 'o', color='red',
         markersize=5, label='Actual', alpha=0.7, zorder=6)
ax2.axvline(x=hist_length, color='green', linestyle='--',
            linewidth=2, alpha=0.7)
ax2.set_xlabel('Time Step', fontsize=11)
ax2.set_ylabel('Value', fontsize=11)
ax2.set_title(f'Tweedie (Trained Params)\nMASE: {mase_tweedie_learned:.4f}',
              fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Tweedie (fixed to Normal's params)
ax3 = axes[2]
ax3.plot(hist_x, ts[:hist_length], 'o-', color='black',
         linewidth=1.5, markersize=3, label='Historical', alpha=0.8)
ax3.plot(forecast_x, median_tweedie_fixed, color='orange',
         linewidth=2.5, label='Forecast (Median)', zorder=5)
ax3.fill_between(forecast_x, q10_tweedie_fixed, q90_tweedie_fixed,
                 alpha=0.25, color='orange', label='80% PI')
ax3.fill_between(forecast_x, q25_tweedie_fixed, q75_tweedie_fixed,
                 alpha=0.4, color='orange', label='50% PI')
ax3.plot(forecast_x, actuals, 'o', color='red',
         markersize=5, label='Actual', alpha=0.7, zorder=6)
ax3.axvline(x=hist_length, color='green', linestyle='--',
            linewidth=2, alpha=0.7)
ax3.set_xlabel('Time Step', fontsize=11)
ax3.set_ylabel('Value', fontsize=11)
ax3.set_title(f'Tweedie (Fixed to Normal)\nMASE: {mase_tweedie_fixed:.4f}',
              fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/TSForecasting/fixed_parameters_diagnostic.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved: fixed_parameters_diagnostic.png")
