"""
Day 4 — Walk-Forward Validation (Expanding Window)

- 1-step ahead prediction
- Expanding training window
- Baselines:
    * Zero-return
    * Linear Regression (lagged returns)
- Metric: R² only
"""
import numpy as np
from sklearn.metrics import r2_score

from return_predictability.src.load_data import load_data
from return_predictability.src.returns import compute_log_returns
from return_predictability.src.features import create_lagged_features
from return_predictability.src.baselines import zero_return_baseline, linear_regression_baseline

# ======================================================
# WALK-FORWARD EXPERIMENT
# ======================================================
