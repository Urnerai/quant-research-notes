from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def zero_return_baseline(
        y:pd.Series
) ->np.ndarray:
    """
    Zero Return Baseline
    Always predicts zero return.
    
    Model
    -----
    r_hat_t = 0

    Parameters
    ----------
    y : pd.Series
        True returns.

    Returns
    -------
    np.ndarray
        Predicted returns (all zeros).
    """
    # --- Validation ---
    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series.")
    return np.zeros(len(y))

def linear_regression_baseline(
        X:pd.DataFrame, y:pd.Series
) -> Tuple[np.ndarray, LinearRegression]:
    """
    Linear regression baseline for return prediction.

    The model assumes a linear relationship between current returns
    and past returns (lags):
    
    Model
    -----
    r_hat_t = beta_0 + sum(beta_k * r_{t-k})

    where r_{t-k} are lagged returns.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix of lagged returns.
    y : pd.Series
        Target return series.

    Returns
    -------
    model : LinearRegression
        Fitted linear regression model.
    y_pred : np.ndarray
        In-sample predictions.
    """
    # --- Validation ---
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")
    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series.")
    if len(X)!=len(y):
        raise ValueError("X and y must have the same length.")
    
    # --- Fit Linear Regression Model ---
    model=LinearRegression()
    model.fit(X,y)
    y_pred=model.predict(X)

    return model, y_pred
    
