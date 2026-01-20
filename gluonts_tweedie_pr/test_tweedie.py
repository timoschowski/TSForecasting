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

import pytest

import numpy as np
import torch

from gluonts.torch.distributions.tweedie import Tweedie, TweedieOutput


@pytest.mark.parametrize("mu", [0.5, 1.0, 5.0])
@pytest.mark.parametrize("phi", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("p", [1.2, 1.5, 1.8])
def test_tweedie_mean_variance(mu, phi, p):
    """Test that mean and variance match theoretical values."""
    dist = Tweedie(mu=mu, phi=phi, p=p)

    # Check mean
    assert torch.allclose(dist.mean, torch.tensor(mu), rtol=1e-5)

    # Check variance: Var(Y) = phi * mu^p
    expected_var = phi * (mu ** p)
    assert torch.allclose(dist.variance, torch.tensor(expected_var), rtol=1e-5)


@pytest.mark.parametrize("mu", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("phi", [0.5, 1.0])
@pytest.mark.parametrize("p", [1.3, 1.5, 1.7])
def test_tweedie_sampling(mu, phi, p):
    """Test that sampling produces valid samples with correct shape."""
    dist = Tweedie(mu=mu, phi=phi, p=p)

    # Sample
    samples = dist.sample((1000,))

    # Check shape
    assert samples.shape == (1000,)

    # Check that samples are non-negative
    assert torch.all(samples >= 0)

    # Check that sample mean is close to mu (with tolerance)
    sample_mean = samples.mean()
    assert torch.abs(sample_mean - mu) < mu * 0.5  # Allow 50% deviation


@pytest.mark.parametrize("mu", [1.0, 2.0])
@pytest.mark.parametrize("phi", [0.5, 1.0])
@pytest.mark.parametrize("p", [1.5])
def test_tweedie_log_prob(mu, phi, p):
    """Test that log_prob is finite and reasonable."""
    dist = Tweedie(mu=mu, phi=phi, p=p)

    # Test with various values including zero
    values = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0])
    log_probs = dist.log_prob(values)

    # Check that log probabilities are finite
    assert torch.all(torch.isfinite(log_probs))

    # For zero values, probability should be positive (some mass at zero)
    assert log_probs[0] > -100  # Not too negative


@pytest.mark.parametrize("mu", [1.0, 5.0])
@pytest.mark.parametrize("phi", [0.5, 1.5])
def test_tweedie_output_domain_map(mu, phi):
    """Test that TweedieOutput domain_map produces positive parameters."""
    output = TweedieOutput(p=1.5)

    # Create raw network outputs (can be negative)
    mu_raw = torch.tensor([[-1.0], [0.0], [1.0]])
    phi_raw = torch.tensor([[-0.5], [0.5], [1.5]])

    # Apply domain map
    mu_mapped, phi_mapped = output.domain_map(mu_raw, phi_raw)

    # Check that all values are positive
    assert torch.all(mu_mapped > 0)
    assert torch.all(phi_mapped > 0)

    # Check shape
    assert mu_mapped.shape == (3,)
    assert phi_mapped.shape == (3,)


@pytest.mark.parametrize("p", [1.2, 1.5, 1.8])
def test_tweedie_output_distribution(p):
    """Test that TweedieOutput creates valid distributions."""
    output = TweedieOutput(p=p)

    mu = torch.tensor([1.0, 2.0, 3.0])
    phi = torch.tensor([0.5, 1.0, 1.5])

    # Create distribution
    dist = output.distribution((mu, phi))

    # Check type
    assert isinstance(dist, Tweedie)

    # Check parameters
    assert torch.allclose(dist.mu, mu)
    assert torch.allclose(dist.phi, phi)
    assert torch.allclose(dist.p, torch.tensor(p))


@pytest.mark.parametrize("p", [1.3, 1.5, 1.7])
def test_tweedie_output_with_scale(p):
    """Test that TweedieOutput correctly applies scale."""
    output = TweedieOutput(p=p)

    mu = torch.tensor([1.0, 2.0])
    phi = torch.tensor([0.5, 1.0])
    scale = torch.tensor([2.0, 3.0])

    # Create distribution with scale
    dist = output.distribution((mu, phi), scale=scale)

    # Check that mu is scaled
    assert torch.allclose(dist.mu, mu * scale)
    assert torch.allclose(dist.phi, phi)


def test_tweedie_batch_shape():
    """Test that Tweedie handles batch shapes correctly."""
    mu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    phi = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    p = 1.5

    dist = Tweedie(mu=mu, phi=phi, p=p)

    # Check batch shape
    assert dist.batch_shape == (2, 2)

    # Sample
    samples = dist.sample()
    assert samples.shape == (2, 2)


def test_tweedie_output_event_shape():
    """Test that TweedieOutput has correct event shape."""
    output = TweedieOutput(p=1.5)
    assert output.event_shape == ()


def test_tweedie_output_value_in_support():
    """Test that TweedieOutput provides a valid value in support."""
    output = TweedieOutput(p=1.5)
    value = output.value_in_support
    assert value >= 0
    assert isinstance(value, float)


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0, 2.5])
def test_tweedie_invalid_p_values(p):
    """Test that invalid p values are rejected."""
    with pytest.raises(AssertionError):
        output = TweedieOutput(p=p)
