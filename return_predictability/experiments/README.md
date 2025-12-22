# Experiments

This directory contains reproducible experiments used to evaluate
daily return predictability under increasingly strict assumptions.

Each experiment is designed to answer a specific research question and is
intentionally simple, transparent, and time-aware.

## baseline_return_predictability.py

Tests whether daily log returns can be predicted using simple baselines.

### Description
- Computes daily log returns
- Constructs lagged return features
- Compares:
  - Zero return baseline
  - Linear regression on lagged returns

### How to Run

From the project root:

```bash
python -m return_predictability.experiments.baseline_return_predictability
```
## walk_forward_return_predictability.py

Evaluates whether any apparent in-sample predictability survives under
realistic, time-aware validation.

### Description

Uses the same return definition and lagged features as the baseline experiment
Applies expanding-window walk-forward validation
At each time step:
-Trains the model using only past data
-Predicts the next day’s return
-Produces a true out-of-sample prediction sequence

### How to Run

From the project root:

```bash
python -m return_predictability.experiments.walk_forward_return_predictability
```