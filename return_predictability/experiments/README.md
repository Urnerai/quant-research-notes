# Experiments

This directory contains reproducible experiments used to evaluate
return predictability under different assumptions.

## baseline_return_predictability.py

Tests whether daily log returns can be predicted using simple baselines.

### Description
- Computes daily log returns
- Constructs lagged return features
- Compares:
  - Zero return baseline
  - Linear regression on lagged returns

### How to run

From the project root:

```bash
python -m return_predictability.experiments.baseline_return_predictability
