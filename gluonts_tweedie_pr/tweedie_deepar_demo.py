#!/usr/bin/env python
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Demonstration of DeepAR with Tweedie distribution on synthetic data.

This script generates a sinusoidal time series with noise and a linear trend,
then trains a DeepAR model with Tweedie distribution output and visualizes
the results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import TweedieOutput
from gluonts.evaluation import make_evaluation_predictions, Evaluator


def generate_synthetic_data(
    n_series=5,
    n_timesteps=200,
    prediction_length=24,
    freq="H",
):
    """
    Generate synthetic time series with sinusoidal pattern, noise, and linear trend.

    Parameters
    ----------
    n_series : int
        Number of time series to generate
    n_timesteps : int
        Total length of each time series
    prediction_length : int
        Length of the prediction horizon
    freq : str
        Frequency string (e.g., "H" for hourly)

    Returns
    -------
    train_ds : ListDataset
        Training dataset
    test_ds : ListDataset
        Test dataset (includes prediction_length more time steps)
    """
    np.random.seed(42)

    train_data = []
    test_data = []

    for i in range(n_series):
        # Time index
        t = np.arange(n_timesteps + prediction_length)

        # Sinusoidal pattern with different frequencies and phases
        period = 24 + np.random.randn() * 2  # Around 24 hours
        phase = np.random.rand() * 2 * np.pi
        amplitude = 2 + np.random.rand() * 3
        sinusoid = amplitude * np.sin(2 * np.pi * t / period + phase)

        # Linear trend
        trend_slope = 0.01 + np.random.rand() * 0.02
        trend = trend_slope * t

        # Base level
        base_level = 5 + np.random.rand() * 5

        # Combine components
        clean_signal = base_level + sinusoid + trend

        # Add noise with some zeros (Tweedie characteristic)
        noise = np.random.gamma(2, 0.5, size=len(t))
        n_zeros = int(len(t) * 0.1)  # 10% zeros
        zero_indices = np.random.choice(len(t), n_zeros, replace=False)
        noise[zero_indices] = 0

        # Final time series (ensure non-negative)
        ts = np.maximum(0, clean_signal + noise - 2)

        # Start date
        start = pd.Timestamp(datetime.now())

        # Training data (exclude last prediction_length points)
        train_data.append({
            "start": start,
            "target": ts[:n_timesteps].tolist(),
        })

        # Test data (include all points)
        test_data.append({
            "start": start,
            "target": ts.tolist(),
        })

    train_ds = ListDataset(train_data, freq=freq)
    test_ds = ListDataset(test_data, freq=freq)

    return train_ds, test_ds


def train_and_evaluate(train_ds, test_ds, prediction_length, distr_output, epochs=20):
    """
    Train DeepAR model and generate predictions.

    Parameters
    ----------
    train_ds : Dataset
        Training dataset
    test_ds : Dataset
        Test dataset
    prediction_length : int
        Prediction horizon length
    distr_output : DistributionOutput
        Distribution output to use
    epochs : int
        Number of training epochs

    Returns
    -------
    forecasts : list
        List of forecast objects
    tss : list
        List of actual time series
    """
    estimator = DeepAREstimator(
        freq="H",
        prediction_length=prediction_length,
        num_layers=2,
        hidden_size=40,
        dropout_rate=0.1,
        distr_output=distr_output,
        batch_size=32,
        num_batches_per_epoch=50,
        trainer_kwargs={
            "max_epochs": epochs,
            "accelerator": "cpu",
            "logger": False,
            "enable_progress_bar": True,
        },
    )

    print(f"\nTraining DeepAR with {distr_output.__class__.__name__}...")
    predictor = estimator.train(train_ds)

    print("\nGenerating predictions...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,
        predictor=predictor,
        num_samples=100,
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss


def plot_forecasts(
    forecasts,
    tss,
    prediction_length,
    n_plots=3,
    save_path="tweedie_deepar_forecast.png",
):
    """
    Plot forecast results.

    Parameters
    ----------
    forecasts : list
        List of forecast objects
    tss : list
        List of actual time series
    prediction_length : int
        Prediction horizon length
    n_plots : int
        Number of time series to plot
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for i, (forecast, ts) in enumerate(zip(forecasts[:n_plots], tss[:n_plots])):
        ax = axes[i]

        # Plot historical data
        ts_data = ts[-prediction_length * 3:]  # Last 3 * prediction_length points
        ax.plot(
            range(len(ts_data)),
            ts_data,
            label="Actual",
            color="black",
            linewidth=2,
        )

        # Forecast start index
        forecast_start = len(ts_data) - prediction_length

        # Plot forecast median
        forecast_index = range(forecast_start, len(ts_data))
        ax.plot(
            forecast_index,
            forecast.median,
            label="Forecast (median)",
            color="blue",
            linewidth=2,
        )

        # Plot prediction intervals
        ax.fill_between(
            forecast_index,
            forecast.quantile(0.1),
            forecast.quantile(0.9),
            alpha=0.3,
            color="blue",
            label="80% prediction interval",
        )

        ax.fill_between(
            forecast_index,
            forecast.quantile(0.25),
            forecast.quantile(0.75),
            alpha=0.5,
            color="blue",
            label="50% prediction interval",
        )

        # Vertical line at forecast start
        ax.axvline(
            x=forecast_start,
            color="red",
            linestyle="--",
            linewidth=1,
            label="Forecast start",
        )

        ax.set_title(f"Time Series {i + 1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 70)
    print("DeepAR with Tweedie Distribution - Demonstration")
    print("=" * 70)

    # Parameters
    n_series = 10
    n_timesteps = 200
    prediction_length = 24
    freq = "H"
    epochs = 20

    # Generate synthetic data
    print("\nGenerating synthetic time series...")
    print(f"  Number of series: {n_series}")
    print(f"  Length per series: {n_timesteps}")
    print(f"  Prediction length: {prediction_length}")
    train_ds, test_ds = generate_synthetic_data(
        n_series=n_series,
        n_timesteps=n_timesteps,
        prediction_length=prediction_length,
        freq=freq,
    )

    # Train with Tweedie distribution
    tweedie_output = TweedieOutput(p=1.5)
    forecasts, tss = train_and_evaluate(
        train_ds=train_ds,
        test_ds=test_ds,
        prediction_length=prediction_length,
        distr_output=tweedie_output,
        epochs=epochs,
    )

    # Evaluate
    print("\nEvaluating forecasts...")
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))

    print("\nAggregate Metrics:")
    print("-" * 50)
    for key, value in agg_metrics.items():
        print(f"{key:30s}: {value:.4f}")

    # Plot results
    print("\nPlotting forecasts...")
    plot_forecasts(
        forecasts=forecasts,
        tss=tss,
        prediction_length=prediction_length,
        n_plots=min(3, n_series),
        save_path="tweedie_deepar_forecast.png",
    )

    print("\n" + "=" * 70)
    print("Demonstration completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
