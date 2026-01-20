#!/usr/bin/env python
"""
Verify what scale the network parameters are for by checking loss computation.
"""

import sys
sys.path.insert(0, '/home/user/gluonts/src')

import torch
from gluonts.torch.distributions import TweedieOutput, NormalOutput

print("="*70)
print("VERIFYING SCALE INTERPRETATION")
print("="*70)

# Scenario: Data with mean=10, scale=10 (so normalized mean=1)
# Target is ORIGINAL scale (mean=10)
target = torch.tensor([10.0])
scale = torch.tensor([10.0])

# Case 1: Network outputs params for NORMALIZED data (mean=1)
print("\nCase 1: Network outputs for normalized data (mu_net=1)")
mu_net = torch.tensor([1.0])
phi_net = torch.tensor([0.5])

tweedie_out = TweedieOutput(p=1.5)
normal_out = NormalOutput()

# Create distributions with scale
tweedie_distr = tweedie_out.distribution((mu_net, phi_net), scale=scale)
normal_distr = normal_out.distribution((mu_net, torch.tensor([0.5])), scale=scale)

print(f"  Tweedie distribution:")
print(f"    mu_net={mu_net.item()}, phi_net={phi_net.item()}, scale={scale.item()}")
print(f"    After scaling: mu={tweedie_distr.mu.item():.2f}, phi={tweedie_distr.phi.item():.4f}")
print(f"    Mean={tweedie_distr.mean.item():.2f}, Var={tweedie_distr.variance.item():.2f}")
print(f"    log_prob(target={target.item()})={tweedie_distr.log_prob(target).item():.4f}")

print(f"\n  Normal distribution (via AffineTransformed):")
print(f"    After scaling: mean={normal_distr.mean.item():.2f}, stddev={normal_distr.stddev.item():.2f}")
print(f"    log_prob(target={target.item()})={normal_distr.log_prob(target).item():.4f}")

# Case 2: Network outputs params for ORIGINAL data (mu_net=10)
print("\n\nCase 2: Network outputs for original data (mu_net=10)")
mu_net2 = torch.tensor([10.0])
phi_net2 = torch.tensor([5.0])  # Adjusted for original scale

tweedie_distr2 = tweedie_out.distribution((mu_net2, phi_net2), scale=scale)
normal_distr2 = normal_out.distribution((mu_net2, torch.tensor([0.5])), scale=scale)

print(f"  Tweedie distribution:")
print(f"    mu_net={mu_net2.item()}, phi_net={phi_net2.item()}, scale={scale.item()}")
print(f"    After scaling: mu={tweedie_distr2.mu.item():.2f}, phi={tweedie_distr2.phi.item():.4f}")
print(f"    Mean={tweedie_distr2.mean.item():.2f}, Var={tweedie_distr2.variance.item():.2f}")
print(f"    log_prob(target={target.item()})={tweedie_distr2.log_prob(target).item():.4f}")

print(f"\n  Normal distribution:")
print(f"    After scaling: mean={normal_distr2.mean.item():.2f}, stddev={normal_distr2.stddev.item():.2f}")
print(f"    log_prob(target={target.item()})={normal_distr2.log_prob(target).item():.4f}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nFor Normal, the AffineTransformed approach clearly expects")
print("params for normalized data (Case 1 makes sense).")
print("\nFor Tweedie, the current implementation with phi * scale^0.5")
print("also expects params for normalized data.")
print("\nThe issue must be elsewhere - possibly in how phi is learned during training")
print("or in the Tweedie distribution's sample() or log_prob() methods.")
