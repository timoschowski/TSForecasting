# Comparison: test_tweedie.py vs test_negative_binomial.py

## Summary

Both distributions override the `distribution()` method to handle scaling differently from the base class (which uses AffineTransformed), but **only Tweedie has unit tests for scaling**.

## Detailed Comparison

### 1. Scaling Implementation

**NegativeBinomial** (lines 85-96 in negative_binomial.py):
```python
def distribution(self, distr_args, loc=None, scale=None):
    total_count, logits = distr_args

    if scale is not None:
        logits += scale.log()  # Scale by adding log(scale) to logits

    return NegativeBinomial(total_count=total_count, logits=logits)
```
- **Comment**: "We cannot scale using affine transformation since negative binomial should return integers"
- **Scaling**: Only logits is scaled (additive in log space)
- **total_count**: NOT scaled

**Tweedie** (lines 248-266 in tweedie.py):
```python
def distribution(self, distr_args, loc=None, scale=None):
    mu, phi = distr_args

    if scale is not None:
        mu = mu * scale               # Scale mu linearly
        phi = phi / scale^(p-1)       # Scale phi by division

    return Tweedie(mu=mu, phi=phi, p=self.p)
```
- **Comment**: Custom scaling to handle zero-inflated nature
- **Scaling**: Both mu and phi are scaled
- **Formula**: `phi / scale^(p-1)` reduces dispersion as scale increases

### 2. Unit Test Coverage

**NegativeBinomial Tests** (test_negative_binomial.py):
```python
# Tests included:
✓ test_custom_neg_bin_logpdf_matches_scipy (compare with scipy)
✓ test_custom_neg_bin_cdf (compare with scipy)
✓ test_custom_neg_bin_icdf (compare with scipy)

# Tests NOT included:
✗ No scaling tests
✗ No test_negative_binomial_output_with_scale
✗ No verification of logits += scale.log() behavior
```

**Tweedie Tests** (test_tweedie.py):
```python
# Tests included:
✓ test_tweedie_mean_variance
✓ test_tweedie_sampling
✓ test_tweedie_log_prob
✓ test_tweedie_output_domain_map
✓ test_tweedie_output_distribution
✓ test_tweedie_output_with_scale ← UNIQUE TO TWEEDIE
✓ test_tweedie_batch_shape
✓ test_tweedie_output_event_shape
✓ test_tweedie_output_value_in_support
✓ test_tweedie_invalid_p_values
```

**Tweedie has MORE comprehensive tests**, including the scaling test that caught the bug!

### 3. Tolerance Levels

**NegativeBinomial**:
```python
assert np.allclose(log_pdf_torch, log_pdf_scipy, rtol=1e-4)  # log prob
assert np.allclose(torch_cdf, scipy_cdf)                      # cdf (default tol)
assert np.allclose(torch_icdf, scipy_icdf)                    # icdf (default tol)
```

**Tweedie**:
```python
assert torch.allclose(dist.mean, torch.tensor(mu), rtol=1e-5)       # mean (stricter)
assert torch.allclose(dist.variance, ..., rtol=1e-5)                # variance (stricter)
assert torch.allclose(dist.mu, mu * scale)                          # mu (default tol)
assert torch.allclose(dist.phi, expected_phi, rtol=1e-4)            # phi scaling
```

**Comparison**:
- Both use `rtol=1e-4` for numerical comparisons (log_prob, phi scaling)
- Tweedie uses stricter `rtol=1e-5` for theoretical properties (mean, variance)
- **Tolerances are similar and appropriate**

### 4. Test Patterns

**Both follow similar patterns**:
- ✓ Parameterized tests with multiple values
- ✓ Check distribution properties
- ✓ Validate domain constraints
- ✓ Test batch shapes
- ✓ Validate edge cases

**Key Difference**:
- **Tweedie tests scaling behavior explicitly**
- NegativeBinomial does NOT test scaling, despite implementing custom scaling logic

## Findings

### 1. Scaling Tests

**Issue**: NegativeBinomial has no unit tests for its custom scaling implementation!

This is actually a **gap** in NegativeBinomial testing. The implementation scales logits
by adding log(scale), but there's no test verifying this works correctly.

**Recommendation**: NegativeBinomial should add a test like:
```python
def test_negative_binomial_output_with_scale():
    """Test that NegativeBinomialOutput correctly applies scale."""
    output = NegativeBinomialOutput()

    total_count = torch.tensor([5.0, 10.0])
    logits = torch.tensor([1.0, 2.0])
    scale = torch.tensor([2.0, 3.0])

    dist = output.distribution((total_count, logits), scale=scale)

    # Check that total_count is unchanged
    assert torch.allclose(dist.total_count, total_count)

    # Check that logits is scaled: logits_new = logits_old + log(scale)
    expected_logits = logits + scale.log()
    assert torch.allclose(dist.logits, expected_logits, rtol=1e-4)
```

### 2. Tolerance Levels

**Tweedie tolerances are appropriate**:
- `rtol=1e-4` matches NegativeBinomial's tolerance for numerical comparisons
- `rtol=1e-5` for mean/variance is reasonable (theoretical properties should be more accurate)
- Default tolerance for parameter equality checks is fine

**No changes needed** to Tweedie tolerances.

### 3. Test Quality

**Tweedie tests are MORE comprehensive** than NegativeBinomial:
- ✓ Tests scaling behavior (which caught the bug!)
- ✓ Tests more edge cases
- ✓ More parameterized tests
- ✓ Better coverage overall

The scaling test `test_tweedie_output_with_scale` was **crucial** for:
1. Catching that the original test expectation was wrong
2. Verifying the implementation matches the intended behavior
3. Ensuring consistency across parameter values

## Conclusion

**Answer to your questions**:

1. **Does NegativeBinomial include scaling tests?**
   - No ❌ - NegativeBinomial has NO scaling tests despite custom scaling implementation
   - This is actually a testing gap in GluonTS

2. **Are tolerance intervals similar?**
   - Yes ✓ - Both use `rtol=1e-4` for numerical comparisons
   - Tweedie is slightly stricter (`rtol=1e-5`) for theoretical properties
   - Tolerances are appropriate and consistent

**Tweedie testing is actually BETTER than NegativeBinomial** - it includes the scaling test
that NegativeBinomial is missing. The fix we made to `test_tweedie_output_with_scale`
ensures it correctly verifies the implementation behavior.

**Recommendation**: Consider adding a similar scaling test for NegativeBinomial.
