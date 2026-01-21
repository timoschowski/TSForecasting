#!/usr/bin/env python
"""
Test parameter recovery with TweedieFull (PyTorch-style implementation).

This tests whether the full log-likelihood with normalizing constant
provides correct gradients, allowing us to recover known parameters.

The deviance-based Tweedie fails this test (phi diverges to 65x true value).
Let's see if TweedieFull passes.
"""

import sys
sys.path.insert(0, '/home/user/gluonts/src')

import numpy as np
import torch

try:
    from tweedie import tweedie as reference_tweedie
    HAS_TWEEDIE = True
except ImportError:
    HAS_TWEEDIE = False
    print("ERROR: tweedie package not installed")
    sys.exit(1)

from gluonts.torch.distributions.tweedie_full import TweedieFull

print("="*80)
print("PARAMETER RECOVERY TEST: TweedieFull (PyTorch-style)")
print("="*80)

# True parameters
true_mu = 2.0
true_phi = 0.5
true_p = 1.5

print(f"\nTrue parameters:")
print(f"  mu = {true_mu}")
print(f"  phi = {true_phi}")
print(f"  p = {true_p}")

# Generate samples from reference implementation
print(f"\nGenerating 100 samples from reference Tweedie...")
ref_dist = reference_tweedie(p=true_p, mu=true_mu, phi=true_phi)
np.random.seed(42)
samples = ref_dist.rvs(size=100)
samples_tensor = torch.tensor(samples, dtype=torch.float32)

print(f"  Sample mean: {samples.mean():.3f} (expected: {true_mu:.3f})")
print(f"  Sample var:  {samples.var():.3f} (expected: {true_phi * true_mu**true_p:.3f})")

# Initialize learnable parameters (with perturbation)
log_mu = torch.nn.Parameter(torch.tensor(np.log(true_mu * 1.2)))
log_phi = torch.nn.Parameter(torch.tensor(np.log(true_phi * 1.3)))

# Optimizer
optimizer = torch.optim.Adam([log_mu, log_phi], lr=0.01)

print(f"\nOptimizing with TweedieFull...")
print(f"  Initial: mu={np.exp(log_mu.item()):.3f}, phi={np.exp(log_phi.item()):.3f}")

# Track progress
losses = []
for step in range(1000):
    optimizer.zero_grad()

    mu = torch.exp(log_mu)
    phi = torch.exp(log_phi)

    try:
        dist = TweedieFull(mu=mu, phi=phi, p=true_p)
        nll = -dist.log_prob(samples_tensor).mean()

        nll.backward()
        optimizer.step()

        losses.append(nll.item())

        if step % 200 == 0:
            print(f"  Step {step:4d}: mu={mu.item():.3f}, phi={phi.item():.3f}, nll={nll.item():.3f}")

    except Exception as e:
        print(f"\n  ERROR at step {step}: {e}")
        print(f"  mu={mu.item():.6f}, phi={phi.item():.6f}")
        break

# Final results
recovered_mu = torch.exp(log_mu).item()
recovered_phi = torch.exp(log_phi).item()

print(f"\n{'='*80}")
print("RESULTS")
print('='*80)

print(f"\nRecovered parameters:")
print(f"  mu:  {recovered_mu:.6f} (true: {true_mu:.6f})")
print(f"  phi: {recovered_phi:.6f} (true: {true_phi:.6f})")

mu_error = abs(recovered_mu - true_mu) / true_mu * 100
phi_error = abs(recovered_phi - true_phi) / true_phi * 100

print(f"\nRelative errors:")
print(f"  mu:  {mu_error:.1f}%")
print(f"  phi: {phi_error:.1f}%")

# Compare with deviance-based result
print(f"\n{'='*80}")
print("COMPARISON WITH DEVIANCE-BASED TWEEDIE")
print('='*80)
print(f"\nDeviance-based Tweedie (from previous tests):")
print(f"  mu:  ~1.893 (error: ~5%)")
print(f"  phi: ~65.398 (error: ~12980%)")
print(f"  → phi diverges to 65x true value (gradient bug)")

print(f"\nTweedieFull (PyTorch-style):")
print(f"  mu:  {recovered_mu:.3f} (error: {mu_error:.1f}%)")
print(f"  phi: {recovered_phi:.3f} (error: {phi_error:.1f}%)")

# Verdict
print(f"\n{'='*80}")
print("VERDICT")
print('='*80)

if mu_error < 20 and phi_error < 20:
    print(f"\n✅ SUCCESS! Both parameters recovered within 20%")
    print(f"   The full log-likelihood with correct gradients WORKS!")
    print(f"   This proves the deviance-based formula was the problem.")
elif mu_error < 20 and phi_error < 100:
    print(f"\n✓ PARTIAL SUCCESS: mu recovers well, phi needs improvement")
    print(f"   Phi error ({phi_error:.1f}%) is better than deviance ({12980:.1f}%)")
    print(f"   May need more optimization steps or tuning")
elif mu_error < 20:
    print(f"\n⚠️ mu recovers but phi still has issues ({phi_error:.1f}%)")
    print(f"   Implementation may still have gradient bugs")
else:
    print(f"\n❌ FAILED: Neither parameter recovers correctly")
    print(f"   Implementation has serious issues")
    print(f"   Check: series expansion, numerical stability, formula correctness")

# Check gradient at true parameters
print(f"\n{'='*80}")
print("GRADIENT CHECK AT TRUE PARAMETERS")
print('='*80)

log_mu_true = torch.nn.Parameter(torch.tensor(np.log(true_mu)))
log_phi_true = torch.nn.Parameter(torch.tensor(np.log(true_phi)))

try:
    dist_true = TweedieFull(mu=torch.exp(log_mu_true), phi=torch.exp(log_phi_true), p=true_p)
    nll_true = -dist_true.log_prob(samples_tensor).mean()
    nll_true.backward()

    print(f"\nAt true parameters:")
    print(f"  NLL = {nll_true.item():.6f}")
    print(f"  grad(log_mu) = {log_mu_true.grad.item():.6f}")
    print(f"  grad(log_phi) = {log_phi_true.grad.item():.6f}")
    print(f"\nExpected: Both gradients should be close to zero")

    if abs(log_mu_true.grad.item()) < 0.1 and abs(log_phi_true.grad.item()) < 0.1:
        print(f"✓ Both gradients are small (correct!)")
    elif abs(log_phi_true.grad.item()) > 0.5:
        print(f"⚠️ grad(log_phi) is large - may still have gradient issues")

    print(f"\nFor comparison, deviance-based Tweedie had:")
    print(f"  grad(log_phi) = -0.604 (large and negative!)")
    print(f"  This pushed phi to increase even at correct value")

except Exception as e:
    print(f"\nERROR computing gradients at true parameters: {e}")
