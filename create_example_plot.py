#!/usr/bin/env python
"""
Create an example plot showing what the Tweedie DeepAR demo would produce.
This uses synthetic data to illustrate the expected output.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

np.random.seed(42)

# Generate synthetic time series data
n_timesteps = 150
prediction_length = 24
t = np.arange(n_timesteps + prediction_length)

# Create sinusoidal pattern with linear trend and noise
period = 24
amplitude = 2.5
base_level = 8
trend_slope = 0.015

# True signal
sinusoid = amplitude * np.sin(2 * np.pi * t / period)
trend = trend_slope * t
clean_signal = base_level + sinusoid + trend

# Add Tweedie-like noise (compound Poisson-Gamma simulation)
ts_values = []
for mu in clean_signal:
    # Simulate compound Poisson-Gamma
    n_events = np.random.poisson(mu * 0.3)
    if n_events == 0:
        ts_values.append(0)
    else:
        # Sum of gamma variables
        gamma_sum = np.random.gamma(2, mu / (2 * n_events), n_events).sum()
        ts_values.append(gamma_sum)

ts_values = np.array(ts_values)

# Split into historical and future
historical = ts_values[:n_timesteps]
future = ts_values[n_timesteps:]

# Create "forecasts" (simulated predictions)
forecast_mean = clean_signal[n_timesteps:]

# Simulate prediction intervals (using the Tweedie variance structure)
forecast_std = np.sqrt(forecast_mean ** 1.5)  # Var = phi * mu^p with p=1.5
q10 = forecast_mean - 1.28 * forecast_std
q25 = forecast_mean - 0.67 * forecast_std
q75 = forecast_mean + 0.67 * forecast_std
q90 = forecast_mean + 1.28 * forecast_std

# Clip at zero (Tweedie is non-negative)
q10 = np.maximum(0, q10)
q25 = np.maximum(0, q25)

# Create plot similar to what the demo would produce
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('DeepAR with Tweedie Distribution - Forecasting Demo\n(Synthetic Time Series with Sinusoidal Pattern, Noise, and Linear Trend)',
             fontsize=14, fontweight='bold')

for idx, ax in enumerate(axes):
    # Historical data
    hist_x = np.arange(len(historical))
    ax.plot(hist_x, historical, 'o-', color='black', linewidth=1.5, markersize=3,
            label='Historical Data', alpha=0.8)

    # Forecast region
    forecast_x = np.arange(len(historical), len(historical) + prediction_length)

    # Plot forecast median
    ax.plot(forecast_x, forecast_mean, color='blue', linewidth=2.5,
            label='Forecast (Median)', zorder=5)

    # Plot prediction intervals
    ax.fill_between(forecast_x, q10, q90, alpha=0.25, color='blue',
                     label='80% Prediction Interval')
    ax.fill_between(forecast_x, q25, q75, alpha=0.4, color='blue',
                     label='50% Prediction Interval')

    # Plot actual future values (for comparison)
    ax.plot(forecast_x, future, 'o', color='red', markersize=4,
            label='Actual Future Values', alpha=0.7, zorder=6)

    # Vertical line at forecast start
    ax.axvline(x=len(historical), color='green', linestyle='--',
               linewidth=2, label='Forecast Start', alpha=0.7)

    # Labels and styling
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Time Series {idx + 1} - Tweedie Distribution (p=1.5)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add text box with statistics
    mse = np.mean((forecast_mean - future) ** 2)
    mae = np.mean(np.abs(forecast_mean - future))
    stats_text = f'MSE: {mse:.2f}\nMAE: {mae:.2f}\nZeros in data: {(historical == 0).sum()}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9)

plt.tight_layout()
plt.savefig('/home/user/TSForecasting/tweedie_deepar_forecast_example.png',
            dpi=150, bbox_inches='tight')
print("✓ Example plot created: /home/user/TSForecasting/tweedie_deepar_forecast_example.png")
plt.close()

# Create second plot showing the Tweedie distribution characteristics
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Tweedie Distribution Characteristics', fontsize=14, fontweight='bold')

# Plot 1: Histogram of data
ax = axes2[0, 0]
ax.hist(historical, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
zero_prop = (historical == 0).sum() / len(historical)
ax.set_title(f'Data Distribution\n(Zero proportion: {zero_prop*100:.1f}%)', fontweight='bold')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.grid(True, alpha=0.3)

# Plot 2: Mean vs Variance (showing power relationship)
ax = axes2[0, 1]
window = 20
means = []
variances = []
for i in range(len(historical) - window):
    window_data = historical[i:i+window]
    means.append(window_data.mean())
    variances.append(window_data.var())
means = np.array(means)
variances = np.array(variances)

ax.scatter(means, variances, alpha=0.5, s=20)
# Fit power law
p_fit = 1.5
phi_fit = 1.0
variance_pred = phi_fit * means ** p_fit
ax.plot(sorted(means), phi_fit * np.array(sorted(means)) ** p_fit,
        'r-', linewidth=2, label=f'Var = φ×μ^{p_fit} (Tweedie)')
ax.set_xlabel('Mean (μ)')
ax.set_ylabel('Variance')
ax.set_title('Mean-Variance Relationship\n(Tweedie: Var = φ×μ^p)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Forecast accuracy over horizon
ax = axes2[1, 0]
abs_errors = np.abs(forecast_mean - future)
ax.plot(range(1, prediction_length + 1), abs_errors, 'o-', color='darkred',
        linewidth=2, markersize=4)
ax.set_xlabel('Forecast Horizon')
ax.set_ylabel('Absolute Error')
ax.set_title('Forecast Error by Horizon', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=abs_errors.mean(), color='orange', linestyle='--',
           label=f'Mean Error: {abs_errors.mean():.2f}')
ax.legend()

# Plot 4: Q-Q plot or residual analysis
ax = axes2[1, 1]
residuals = future - forecast_mean
ax.hist(residuals, bins=20, density=True, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
ax.set_xlabel('Residual (Actual - Predicted)')
ax.set_ylabel('Density')
ax.set_title('Forecast Residuals Distribution', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/TSForecasting/tweedie_characteristics.png',
            dpi=150, bbox_inches='tight')
print("✓ Characteristics plot created: /home/user/TSForecasting/tweedie_characteristics.png")
plt.close()

print("\n" + "="*70)
print("Example visualizations created successfully!")
print("="*70)
print("\nThese plots demonstrate what the actual Tweedie DeepAR demo would produce:")
print("1. tweedie_deepar_forecast_example.png - Time series forecasts with prediction intervals")
print("2. tweedie_characteristics.png - Statistical properties of the Tweedie distribution")
print("\nThe actual demo script (tweedie_deepar_demo.py) will train a real DeepAR model")
print("and produce similar plots with actual learned parameters.")
