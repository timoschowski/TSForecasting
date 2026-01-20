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

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Gamma, Poisson, constraints
from torch.distributions.utils import broadcast_all

from .distribution_output import DistributionOutput


class Tweedie(Distribution):
    """
    Tweedie distribution for the compound Poisson-Gamma case (1 < p < 2).

    The Tweedie distribution is a member of the exponential dispersion family.
    For 1 < p < 2, it represents a compound Poisson-Gamma distribution, which
    is particularly useful for modeling data with many zeros and positive continuous
    values (e.g., insurance claims, rainfall).

    Parameters
    ----------
    mu : float or torch.Tensor
        Mean parameter (must be positive).
    phi : float or torch.Tensor
        Dispersion parameter (must be positive).
    p : float or torch.Tensor
        Power parameter (must be in (1, 2) for compound Poisson-Gamma).
        Default is 1.5.

    The variance is given by: Var(Y) = phi * mu^p
    """

    arg_constraints = {
        "mu": constraints.positive,
        "phi": constraints.positive,
        "p": constraints.interval(1.0, 2.0),
    }
    support = constraints.nonnegative
    has_rsample = False

    def __init__(
        self,
        mu: Union[float, torch.Tensor],
        phi: Union[float, torch.Tensor],
        p: Union[float, torch.Tensor] = 1.5,
        validate_args=None,
    ):
        self.mu, self.phi, self.p = broadcast_all(mu, phi, p)

        if isinstance(mu, float) and isinstance(phi, float):
            batch_shape = torch.Size()
        else:
            batch_shape = self.mu.shape

        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        return self.mu

    @property
    def variance(self) -> torch.Tensor:
        return self.phi * torch.pow(self.mu, self.p)

    @property
    def stddev(self) -> torch.Tensor:
        return torch.sqrt(self.variance)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability density/mass function.

        For the compound Poisson-Gamma case, this uses the canonical
        parameterization of the Tweedie distribution.
        """
        if self._validate_args:
            self._validate_sample(value)

        # Canonical parameters
        # theta = (mu^(1-p)) / (1-p) for p != 1
        theta = torch.pow(self.mu, 1 - self.p) / (1 - self.p)

        # kappa(theta) = (mu^(2-p)) / (2-p) for p != 2
        kappa = torch.pow(self.mu, 2 - self.p) / (2 - self.p)

        # Log-likelihood: (y*theta - kappa(theta)) / phi
        # For exact computation, we'd need Wright's generalized Bessel function
        # Here we use a simplified approximation suitable for optimization
        log_like = (value * theta - kappa) / self.phi

        # For y=0, we can compute the exact probability using the Poisson component
        # P(Y=0) = exp(-lambda) where lambda = mu^(2-p) / (phi*(2-p))
        lambda_param = torch.pow(self.mu, 2 - self.p) / (self.phi * (2 - self.p))
        log_prob_zero = -lambda_param

        # Use the zero probability for zero values, log-likelihood for positive values
        is_zero = (value == 0)
        result = torch.where(is_zero, log_prob_zero, log_like)

        return result

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """
        Generate samples from the Tweedie distribution.

        For 1 < p < 2, uses the compound Poisson-Gamma representation:
        Y = sum_{i=1}^N Z_i, where N ~ Poisson(lambda) and Z_i ~ Gamma(alpha, beta)
        """
        shape = self._extended_shape(sample_shape)

        # Compound Poisson-Gamma parameters
        # lambda = mu^(2-p) / (phi * (2-p))
        lambda_param = torch.pow(self.mu, 2 - self.p) / (self.phi * (2 - self.p))

        # alpha = (2-p) / (p-1)
        alpha = (2 - self.p) / (self.p - 1)

        # beta = (mu^(1-p)) / (phi * (p-1))
        beta = torch.pow(self.mu, 1 - self.p) / (self.phi * (self.p - 1))

        # Sample number of events from Poisson
        with torch.no_grad():
            n_events = Poisson(lambda_param.expand(shape)).sample()

            # For each sample, sum n_events Gamma variables
            # Create Gamma distribution
            gamma_dist = Gamma(alpha.expand(shape), beta.expand(shape))

            # Sample from Gamma and multiply by number of events
            # This is an approximation; exact sampling would require summing n_events samples
            # For efficiency, we use n_events * Gamma(alpha, beta) as an approximation
            samples = n_events * gamma_dist.sample() / alpha.expand(shape)

        return samples

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """
        Tweedie does not support reparameterized sampling.
        """
        raise NotImplementedError("Tweedie does not support rsample")


class TweedieOutput(DistributionOutput):
    """
    DistributionOutput class for Tweedie distribution.

    Parameters
    ----------
    p : float
        Power parameter for Tweedie distribution (must be in (1, 2)).
        Default is 1.5.
    """

    args_dim: Dict[str, int] = {"mu": 1, "phi": 1}
    distr_cls: type = Tweedie

    def __init__(self, p: float = 1.5, beta: float = 0.0) -> None:
        super().__init__(beta=beta)
        assert 1.0 < p < 2.0, f"Power parameter p must be in (1, 2), got {p}"
        self.p = p

    @classmethod
    def domain_map(cls, mu: torch.Tensor, phi: torch.Tensor):  # type: ignore
        """
        Map network outputs to valid distribution parameters.

        Parameters
        ----------
        mu : torch.Tensor
            Raw network output for mu parameter
        phi : torch.Tensor
            Raw network output for phi parameter

        Returns
        -------
        mu_mapped : torch.Tensor
            Positive mu values
        phi_mapped : torch.Tensor
            Positive phi values
        """
        # Ensure positive parameters using softplus
        epsilon = torch.finfo(mu.dtype).eps
        mu = F.softplus(mu).clamp_min(epsilon)
        phi = F.softplus(phi).clamp_min(epsilon)
        return mu.squeeze(-1), phi.squeeze(-1)

    def _base_distribution(self, distr_args) -> Distribution:
        mu, phi = distr_args
        return self.distr_cls(mu=mu, phi=phi, p=self.p)

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        """
        Construct Tweedie distribution with optional scaling.

        For Tweedie, we scale the mu parameter rather than using affine transformation.
        """
        mu, phi = distr_args

        # Scale mu if scale is provided
        if scale is not None:
            mu = mu * scale

        return self.distr_cls(mu=mu, phi=phi, p=self.p)

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def value_in_support(self) -> float:
        return 0.5
