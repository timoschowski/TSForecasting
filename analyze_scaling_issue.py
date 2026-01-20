#!/usr/bin/env python
"""
Analyze the scaling issue in Tweedie distribution.
Compare how Normal vs Tweedie handle the scale parameter in DeepAR.
"""

import sys
sys.path.insert(0, '/home/user/gluonts/src')

import torch
import torch.nn.functional as F
from gluonts.torch.distributions import NormalOutput, TweedieOutput

print("="*70)
print("ANALYZING SCALE PARAMETER HANDLING")
print("="*70)

# Simulate what the network outputs (raw values before domain_map)
raw_param1 = torch.tensor([1.0])  # For Normal: loc, For Tweedie: mu
raw_param2 = torch.tensor([0.0])  # For Normal: scale, For Tweedie: phi

print("\n1. RAW NETWORK OUTPUTS (before domain_map):")
print(f"   Param 1 (loc/mu): {raw_param1.item():.4f}")
print(f"   Param 2 (scale/phi): {raw_param2.item():.4f}")

# Apply domain_map for Normal
normal_output = NormalOutput()
loc_normal, scale_normal = normal_output.domain_map(raw_param1.clone(), raw_param2.clone())

print("\n2. NORMAL DISTRIBUTION - After domain_map:")
print(f"   loc: {loc_normal.item():.4f}")
print(f"   scale: {scale_normal.item():.4f} (softplus({raw_param2.item()}) = {F.softplus(raw_param2).item():.4f})")

# Apply domain_map for Tweedie
tweedie_output = TweedieOutput(p=1.5)
mu_tweedie, phi_tweedie = tweedie_output.domain_map(raw_param1.clone(), raw_param2.clone())

print("\n3. TWEEDIE DISTRIBUTION - After domain_map:")
print(f"   mu: {mu_tweedie.item():.4f}")
print(f"   phi: {phi_tweedie.item():.4f} (softplus({raw_param2.item()}) = {F.softplus(raw_param2).item():.4f})")

# Now test with scale parameter (what DeepAR provides based on training data statistics)
scale_factor = torch.tensor([10.0])  # Typical scale from training data

print("\n" + "="*70)
print(f"APPLYING SCALE FACTOR: {scale_factor.item()}")
print("="*70)

# Normal: Uses AffineTransformed internally, which does Y = loc + scale_factor * Z
# where Z ~ Normal(loc_normal, scale_normal)
# So effectively: Y ~ Normal(loc + scale_factor * loc_normal, scale_factor * scale_normal)
print("\n4. NORMAL - How scale is applied:")
print("   Uses AffineTransformed: Y = loc + scale_factor * Z")
print(f"   If loc=None in distribution(), then:")
print(f"   Y ~ Normal(loc={loc_normal.item():.4f}, scale={scale_normal.item():.4f})")
print(f"   is transformed to Y ~ AffineTransformed with scale={scale_factor.item()}")
print(f"   Final effective: Y ~ Normal(0 + {scale_factor.item()}*{loc_normal.item()}, {scale_factor.item()}*{scale_normal.item()})")
print(f"                      = Normal({scale_factor.item()*loc_normal.item():.4f}, {scale_factor.item()*scale_normal.item():.4f})")

# Tweedie: Custom scaling in distribution() method
distr_args_tweedie = (mu_tweedie, phi_tweedie)
distr_tweedie_no_scale = tweedie_output.distribution(distr_args_tweedie, loc=None, scale=None)
distr_tweedie_with_scale = tweedie_output.distribution(distr_args_tweedie, loc=None, scale=scale_factor)

print("\n5. TWEEDIE - How scale is applied in distribution():")
print("   Custom logic in TweedieOutput.distribution():")
print(f"   Original mu: {mu_tweedie.item():.4f}, phi: {phi_tweedie.item():.4f}")
print(f"   After scaling:")
print(f"     mu_scaled = mu * scale = {mu_tweedie.item():.4f} * {scale_factor.item():.4f} = {distr_tweedie_with_scale.mu.item():.4f}")
print(f"     phi_scaled = phi / scale^(p-1) = {phi_tweedie.item():.4f} / {scale_factor.item():.4f}^0.5 = {distr_tweedie_with_scale.phi.item():.4f}")

print("\n" + "="*70)
print("IDENTIFYING THE PROBLEM")
print("="*70)

print("\nThe issue is in Tweedie's distribution() method (lines 249-257 in tweedie.py):")
print("\n  Line 251: mu = mu * scale  ✓ (Correct)")
print(f"  Line 257: phi = phi / scale^(p-1)  ✗ (PROBLEMATIC)")

print("\nWhy this is wrong:")
print("1. For Normal, 'scale' is applied as an affine transformation: Y = scale * Z")
print("   This means the network learns parameters for Z, and scale adjusts to data scale.")

print("\n2. For Tweedie, 'scale' modifies the distribution parameters directly.")
print("   But the formula phi = phi / scale^(p-1) is TOO AGGRESSIVE.")
print(f"   With scale={scale_factor.item()}, it divides phi by {torch.pow(scale_factor, 0.5).item():.4f}")

print("\n3. Result: The learned phi values become incorrect!")
print("   - Network learns phi for normalized data")
print("   - phi gets heavily reduced during scaling")
print("   - This causes the distribution to have wrong variance")
print("   - Forecasts are off-scale (too high/low)")

print("\n" + "="*70)
print("PROPOSED FIX")
print("="*70)

print("\nOption 1: Remove the phi scaling entirely")
print("  Just scale mu, keep phi unchanged:")
print("  - mu_scaled = mu * scale")
print("  - phi_scaled = phi  (NO SCALING)")

print("\nOption 2: Use standard formula phi_scaled = phi * scale^(2-p)")
print("  This preserves variance relationship: Var[s*Y] = s^2 * Var[Y]")
print(f"  - phi_scaled = phi * scale^(2-p) = phi * {scale_factor.item()}^0.5")

print("\nOption 3: Don't scale in distribution(), use AffineTransformed")
print("  Like Normal does - let the base class handle it")
print("  But this may not work well for Tweedie due to zero-inflation")

print("\n" + "="*70)
print("TESTING WITH ACTUAL LEARNED VALUES")
print("="*70)

# Simulate what network might learn for normalized data (mean ~1)
print("\nScenario: Training data has mean=10, normalized to mean=1")
print("Network learns for normalized data:")

raw_mu_learned = torch.tensor([0.5])  # Before softplus
raw_phi_learned = torch.tensor([-1.0])  # Before softplus

mu_learned, phi_learned = tweedie_output.domain_map(raw_mu_learned, raw_phi_learned)
print(f"  mu (after domain_map): {mu_learned.item():.4f}")
print(f"  phi (after domain_map): {phi_learned.item():.4f}")

# Apply scale=10 (to go back to original scale)
scale = torch.tensor([10.0])
distr_scaled = tweedie_output.distribution((mu_learned, phi_learned), loc=None, scale=scale)

print(f"\nAfter applying scale={scale.item()}:")
print(f"  mu_scaled: {distr_scaled.mu.item():.4f}")
print(f"  phi_scaled: {distr_scaled.phi.item():.4f}")
print(f"  Expected mean: {distr_scaled.mean.item():.4f}")
print(f"  Variance: {distr_scaled.variance.item():.4f}")

print("\n✗ PROBLEM: These values don't match what we'd expect!")
print("  The phi scaling by 1/scale^0.5 ≈ 1/3.16 is causing issues.")
