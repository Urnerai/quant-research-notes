from typing import Iterable

import pandas as pd


def create_lagged_features(
        returns: pd.Series,
        lags: Iterable[int]
        
) -> pd.DataFrame:
    """
    Create lagged features for a given time series.
    Parameters
    ----------
    returns : pd.Series
        Time indexed series for which to create lagged features.
    lags : Iterable[int]
        Lag values to include as features.
    Returns
    -------
    pd.DataFrame
        DataFrame with lagged features.
        NaN values are intentionally preserved.
    """
    # --- Validation ---
    if not isinstance(returns, pd.Series):
        raise ValueError("Input data must be a pandas Series.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex.")

    # --- Feature Container ---
    features=pd.DataFrame(index=returns.index)

    # --- Create Lagged Features ---
    for lag in lags:
        if lag <= 0:
            raise ValueError("Lag values must be positive integers.")
        features[f"lag_{lag}"]=returns.shift(lag)
    return features
