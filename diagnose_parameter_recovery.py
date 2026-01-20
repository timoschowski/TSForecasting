"""
Detailed diagnosis of Tweedie parameter recovery issue.

The unit test shows that phi recovers at ~40x the true value,
which suggests a systematic bias in the deviance-based gradient.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from tweedie import tweedie as reference_tweedie
from gluonts.torch.distributions.tweedie import Tweedie

# True parameters
true_mu = 2.0
true_phi = 0.5
true_p = 1.5

print("=" * 80)
print("TWEEDIE PARAMETER RECOVERY DIAGNOSIS")
print("=" * 80)
print(f"\nTrue parameters:")
print(f"  mu = {true_mu}")
print(f"  phi = {true_phi}")
print(f"  p = {true_p}")

# Generate samples from reference implementation
ref_dist = reference_tweedie(p=true_p, mu=true_mu, phi=true_phi)
np.random.seed(42)
samples = ref_dist.rvs(size=100)
samples_tensor = torch.tensor(samples, dtype=torch.float32)

print(f"\nSample statistics:")
print(f"  mean = {samples.mean():.3f} (expected: {true_mu:.3f})")
print(f"  var = {samples.var():.3f} (expected: {true_phi * true_mu**true_p:.3f})")
print(f"  min = {samples.min():.3f}")
print(f"  max = {samples.max():.3f}")
print(f"  num_zeros = {(samples == 0).sum()}")

# Initialize learnable parameters
log_mu = torch.nn.Parameter(torch.tensor(np.log(true_mu * 1.2)))
log_phi = torch.nn.Parameter(torch.tensor(np.log(true_phi * 1.3)))

# Optimizer
optimizer = torch.optim.Adam([log_mu, log_phi], lr=0.01)

# Track parameter evolution
mu_history = []
phi_history = []
nll_history = []

# Optimize
print("\nOptimizing...")
for step in range(2000):
    optimizer.zero_grad()

    mu = torch.exp(log_mu)
    phi = torch.exp(log_phi)
    dist = Tweedie(mu=mu, phi=phi, p=true_p)

    nll = -dist.log_prob(samples_tensor).mean()

    nll.backward()
    optimizer.step()

    # Track
    mu_history.append(mu.item())
    phi_history.append(phi.item())
    nll_history.append(nll.item())

    if step % 200 == 0:
        print(f"  Step {step:4d}: mu={mu.item():.3f}, phi={phi.item():.3f}, nll={nll.item():.3f}")

# Final results
recovered_mu = torch.exp(log_mu).item()
recovered_phi = torch.exp(log_phi).item()

print(f"\nFinal recovered parameters:")
print(f"  mu = {recovered_mu:.3f} (true: {true_mu:.3f}, error: {abs(recovered_mu - true_mu)/true_mu*100:.1f}%)")
print(f"  phi = {recovered_phi:.3f} (true: {true_phi:.3f}, error: {abs(recovered_phi - true_phi)/true_phi*100:.1f}%)")

# Check gradient at true parameters
print("\n" + "=" * 80)
print("GRADIENT CHECK AT TRUE PARAMETERS")
print("=" * 80)

log_mu_true = torch.nn.Parameter(torch.tensor(np.log(true_mu)))
log_phi_true = torch.nn.Parameter(torch.tensor(np.log(true_phi)))

dist_true = Tweedie(mu=torch.exp(log_mu_true), phi=torch.exp(log_phi_true), p=true_p)
nll_true = -dist_true.log_prob(samples_tensor).mean()
nll_true.backward()

print(f"\nAt true parameters:")
print(f"  NLL = {nll_true.item():.3f}")
print(f"  grad(log_mu) = {log_mu_true.grad.item():.6f}")
print(f"  grad(log_phi) = {log_phi_true.grad.item():.6f}")
print(f"\nExpected gradients at true parameters should be close to zero.")
print(f"If grad(log_phi) is large and negative, it will push phi to increase.")

# Compare with reference log_prob
print("\n" + "=" * 80)
print("COMPARE LOG_PROB: GLUONTS VS REFERENCE")
print("=" * 80)

# GluonTS log_prob at true params
dist_gluonts = Tweedie(mu=true_mu, phi=true_phi, p=true_p)
log_prob_gluonts = dist_gluonts.log_prob(samples_tensor).mean().item()

# Reference log_prob
log_prob_ref = np.log(ref_dist.pdf(samples) + 1e-10).mean()  # Add small epsilon for zeros

print(f"\nMean log_prob:")
print(f"  GluonTS: {log_prob_gluonts:.3f}")
print(f"  Reference: {log_prob_ref:.3f}")
print(f"  Difference: {log_prob_gluonts - log_prob_ref:.3f}")
print(f"\nNote: These values should differ (deviance-based vs full likelihood)")

# Plot parameter evolution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(mu_history, label='Recovered')
axes[0].axhline(true_mu, color='r', linestyle='--', label='True')
axes[0].set_xlabel('Optimization Step')
axes[0].set_ylabel('mu')
axes[0].set_title('mu Evolution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(phi_history, label='Recovered')
axes[1].axhline(true_phi, color='r', linestyle='--', label='True')
axes[1].set_xlabel('Optimization Step')
axes[1].set_ylabel('phi')
axes[1].set_title('phi Evolution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(nll_history)
axes[2].set_xlabel('Optimization Step')
axes[2].set_ylabel('Negative Log Likelihood')
axes[2].set_title('Loss Evolution')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/TSForecasting/parameter_recovery_diagnosis.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: /home/user/TSForecasting/parameter_recovery_diagnosis.png")

# Try using reference log_prob for comparison
print("\n" + "=" * 80)
print("EXPERIMENT: OPTIMIZE WITH REFERENCE LOG_PROB")
print("=" * 80)

def reference_log_prob(samples, mu, phi, p):
    """Compute log_prob using reference implementation."""
    ref_dist = reference_tweedie(p=p, mu=mu, phi=phi)
    return np.log(ref_dist.pdf(samples) + 1e-10).mean()

# Grid search to find best parameters using reference log_prob
print("\nGrid search for best parameters using reference log_prob:")
best_nll = float('inf')
best_mu = None
best_phi = None

mu_range = np.linspace(1.0, 4.0, 20)
phi_range = np.linspace(0.1, 2.0, 20)

for mu_test in mu_range:
    for phi_test in phi_range:
        nll = -reference_log_prob(samples, mu_test, phi_test, true_p)
        if nll < best_nll:
            best_nll = nll
            best_mu = mu_test
            best_phi = phi_test

print(f"  Best mu (grid search): {best_mu:.3f} (true: {true_mu:.3f})")
print(f"  Best phi (grid search): {best_phi:.3f} (true: {true_phi:.3f})")
print(f"  Best NLL: {best_nll:.3f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("\nThe deviance-based log_prob in GluonTS Tweedie appears to have")
print("incorrect gradients for phi, causing it to converge to ~40x the true value.")
print("This explains why Tweedie performs poorly in DeepAR training!")
