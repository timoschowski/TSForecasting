# Tweedie Scaling Bug Analysis

## Summary

**Found the bug!** The Tweedie distribution parameter scaling formula is incorrect in `tweedie.py` line 257.

**Current (WRONG):** `phi_scaled = phi / scale^(p-1)`
**Correct:** `phi_scaled = phi * scale^(2-p)`

For p=1.5 and scale=10:
- Current: phi / 10^0.5 = phi / 3.16 (DIVIDES by ~3)
- Correct: phi * 10^0.5 = phi * 3.16 (MULTIPLIES by ~3)

**This is a factor of scale^1.0 = 10x difference!** No wonder the forecasts are completely wrong.

## How DeepAR Uses Scale

From analyzing `/home/user/gluonts/src/gluonts/torch/model/deepar/module.py`:

1. **Compute scale** (line 232): `scale = MeanScaler(training_data)`
   - For data with mean 10, scale ≈ 10

2. **Normalize data** (lines 237, 240): `normalized_data = data / scale`
   - Data with mean 10 becomes data with mean 1

3. **Network learns parameters** for normalized data:
   - Network outputs raw values for mu and phi
   - domain_map() transforms them to valid parameters: mu_net, phi_net
   - These parameters fit the **normalized** data (mean ~1)

4. **Create distribution** (line 360): `distr_output.distribution(params, scale=scale)`
   - Passes scale to convert parameters back to original data scale

## Mathematical Derivation

### For Normal Distribution (works correctly):
- Normalized: Z ~ Normal(mu_net, sigma_net)
- Original: Y = scale * Z
- Therefore: Y ~ Normal(scale * mu_net, scale * sigma_net)
- Implemented via AffineTransformed ✓

### For Tweedie Distribution (current implementation is WRONG):

**Given:**
- Tweedie variance: Var[Y] = φ * μ^p
- Network learns: mu_net, phi_net for normalized data (data/scale)
- Need to find: mu_scaled, phi_scaled for original data

**Derivation:**
- Original data: Y = scale * Z, where Z is normalized data
- E[Y] = scale * E[Z] = scale * mu_net
  - **Therefore: mu_scaled = scale * mu_net** ✓ (Current code is correct)

- Var[Y] = scale^2 * Var[Z] = scale^2 * phi_net * mu_net^p
- But also: Var[Y] = phi_scaled * mu_scaled^p = phi_scaled * (scale * mu_net)^p
- Equating: phi_scaled * scale^p * mu_net^p = scale^2 * phi_net * mu_net^p
- **Therefore: phi_scaled = phi_net * scale^(2-p)** ✗ (Current code is WRONG!)

**For p=1.5:**
- Correct: phi_scaled = phi_net * scale^0.5
- Current: phi_scaled = phi_net / scale^0.5 = phi_net * scale^(-0.5)

**The sign is flipped!** Current code uses scale^(-(p-1)) instead of scale^(2-p).

## Why This Causes the Problem

### Example with scale=10, p=1.5:

**Network learns for normalized data (mean=1):**
- mu_net = 1.0
- phi_net = 0.5

**Current (WRONG) scaling:**
- mu_scaled = 1.0 * 10 = 10 ✓
- phi_scaled = 0.5 / 10^0.5 = 0.5 / 3.16 = **0.158** ✗
- Var_current = 0.158 * 10^1.5 = 5.0
- Expected variance should be: scale^2 * phi_net * mu_net^p = 100 * 0.5 * 1 = 50
- **Off by 10x!** (factor of scale)

**Correct scaling:**
- mu_scaled = 1.0 * 10 = 10 ✓
- phi_scaled = 0.5 * 10^0.5 = 0.5 * 3.16 = **1.58** ✓
- Var_correct = 1.58 * 10^1.5 = 50 ✓

## The Fix

In `/home/user/gluonts/src/gluonts/torch/distributions/tweedie.py`, line 257:

**Change from:**
```python
phi = phi / torch.pow(scale, self.p - 1)
```

**To:**
```python
phi = phi * torch.pow(scale, 2 - self.p)
```

## Impact

This single-line fix should:
1. Make Tweedie forecasts have the correct scale
2. Make Tweedie achieve similar MASE to Normal on clean data
3. Fix the parameter learning issues we observed

The diagnostic test showed:
- Tweedie (trained): MASE 11.68 (completely wrong scale)
- Tweedie (fixed params): MASE 2.20 (nearly perfect)

With this fix, trained Tweedie should achieve ~2.20 MASE instead of 11.68!
