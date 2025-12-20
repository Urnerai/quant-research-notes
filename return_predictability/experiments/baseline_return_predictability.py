import numpy as np

from sklearn.metrics import r2_score

from return_predictability.src.load_data import load_data
from return_predictability.src.returns import compute_log_returns
from return_predictability.src.features import create_lagged_features
from return_predictability.src.baselines import zero_return_baseline, linear_regression_baseline

def main():
    # --- Load Data ---
    data=load_data("tsla.csv")
    
    # --- Compute Returns ---
    returns=compute_log_returns(data["price"])
    
    # --- Create Lagged Features ---
    lags=[1,5]
    X=create_lagged_features(returns, lags)
    y=returns
    
    # --- Clean NaN Values ---
    mask=X.notna().all(axis=1)
    X_clean=X[mask]
    y_clean=y[mask]
    
    # --- Zero Return Baseline ---
    y_pred_zero=zero_return_baseline(y_clean)
    r2_zero=r2_score(y_clean, y_pred_zero)
    
    # --- Linear Regression Baseline ---
    model, y_pred_lr=linear_regression_baseline(X_clean, y_clean)
    r2_lr=r2_score(y_clean, y_pred_lr)

    #--- Results ---
    print(f"Zero Return Baseline R^2: {r2_zero:.4f}")
    print(f"Linear Regression Baseline R^2: {r2_lr:.4f}")

if __name__ == "__main__":
    main()