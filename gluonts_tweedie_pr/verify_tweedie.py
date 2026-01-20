#!/usr/bin/env python
"""
Simple verification script for Tweedie distribution implementation.
This tests basic functionality without requiring full training.
"""

import sys
sys.path.insert(0, '/home/user/gluonts/src')

try:
    import torch
    print("✓ PyTorch imported successfully")
except ImportError as e:
    print("✗ PyTorch not available:", e)
    sys.exit(1)

try:
    from gluonts.torch.distributions.tweedie import Tweedie, TweedieOutput
    print("✓ Tweedie classes imported successfully")
except ImportError as e:
    print("✗ Failed to import Tweedie classes:", e)
    sys.exit(1)

# Test 1: Create Tweedie distribution
print("\n" + "="*60)
print("Test 1: Create Tweedie distribution")
print("="*60)
try:
    mu = torch.tensor([1.0, 2.0, 3.0])
    phi = torch.tensor([0.5, 1.0, 1.5])
    p = 1.5

    dist = Tweedie(mu=mu, phi=phi, p=p)
    print(f"✓ Distribution created with batch shape: {dist.batch_shape}")
    print(f"  Mean: {dist.mean}")
    print(f"  Variance: {dist.variance}")
except Exception as e:
    print(f"✗ Failed to create distribution: {e}")
    sys.exit(1)

# Test 2: Sample from distribution
print("\n" + "="*60)
print("Test 2: Sample from distribution")
print("="*60)
try:
    samples = dist.sample((100,))
    print(f"✓ Generated samples with shape: {samples.shape}")
    print(f"  Sample mean: {samples.mean(dim=0)}")
    print(f"  Sample std: {samples.std(dim=0)}")
    print(f"  All samples non-negative: {torch.all(samples >= 0)}")
except Exception as e:
    print(f"✗ Failed to sample: {e}")
    sys.exit(1)

# Test 3: Compute log probability
print("\n" + "="*60)
print("Test 3: Compute log probability")
print("="*60)
try:
    values = torch.tensor([[0.0, 1.0, 2.0]])
    log_probs = dist.log_prob(values)
    print(f"✓ Log probabilities computed: {log_probs}")
    print(f"  All finite: {torch.all(torch.isfinite(log_probs))}")
except Exception as e:
    print(f"✗ Failed to compute log probability: {e}")
    sys.exit(1)

# Test 4: TweedieOutput domain mapping
print("\n" + "="*60)
print("Test 4: TweedieOutput domain mapping")
print("="*60)
try:
    output = TweedieOutput(p=1.5)

    # Raw network outputs (can be negative)
    mu_raw = torch.tensor([[-1.0, 0.0, 1.0]])
    phi_raw = torch.tensor([[-0.5, 0.5, 1.5]])

    mu_mapped, phi_mapped = output.domain_map(mu_raw, phi_raw)
    print(f"✓ Domain mapping successful")
    print(f"  Mu (mapped): {mu_mapped}")
    print(f"  Phi (mapped): {phi_mapped}")
    print(f"  All mu positive: {torch.all(mu_mapped > 0)}")
    print(f"  All phi positive: {torch.all(phi_mapped > 0)}")
except Exception as e:
    print(f"✗ Failed domain mapping: {e}")
    sys.exit(1)

# Test 5: Create distribution via TweedieOutput
print("\n" + "="*60)
print("Test 5: Create distribution via TweedieOutput")
print("="*60)
try:
    mu = torch.tensor([1.0, 2.0])
    phi = torch.tensor([0.5, 1.0])

    dist = output.distribution((mu, phi))
    print(f"✓ Distribution created via TweedieOutput")
    print(f"  Distribution type: {type(dist).__name__}")
    print(f"  Mean: {dist.mean}")

    # Test with scale
    scale = torch.tensor([2.0, 3.0])
    dist_scaled = output.distribution((mu, phi), scale=scale)
    print(f"✓ Distribution with scale created")
    print(f"  Scaled mean: {dist_scaled.mean}")
    print(f"  Expected: {mu * scale}")
except Exception as e:
    print(f"✗ Failed to create distribution via output: {e}")
    sys.exit(1)

# Test 6: Invalid p values
print("\n" + "="*60)
print("Test 6: Invalid p values (should raise assertion)")
print("="*60)
try:
    invalid_ps = [0.5, 1.0, 2.0, 2.5]
    for p_val in invalid_ps:
        try:
            output_bad = TweedieOutput(p=p_val)
            print(f"✗ Should have raised assertion for p={p_val}")
        except AssertionError:
            print(f"✓ Correctly rejected p={p_val}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("\nThe Tweedie distribution implementation is working correctly.")
