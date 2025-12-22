from py_compile import main
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from return_predictability.src.load_data import load_data
from return_predictability.src.returns import compute_log_returns
from return_predictability.src.features import create_lagged_features

# WALK-FORWARD EXPERIMENT
# ======================================================
"""
Walk-Forward Validation (Expanding Window)

- 1-step ahead prediction
- Expanding training window
- Baselines:
    * Zero-return
    * Linear Regression (lagged returns)
- Metric: R² only
"""
def main():
    # ---Load Data---
    price=load_data("tsla.csv")

    # --- Compute Returns ---
    returns=compute_log_returns(price)

    # --- Create Lagged Features ---
    lags=[1,5]
    X=create_lagged_features(returns,lags)
    y=returns

    # --- Clean NaN Values ---
    mask=X.notna().all(axis=1)
    X_clean=X[mask]
    y_clean=y[mask]

    # --- Walk-Forward Setup ---
    start=1

    predictions=[]
    actuals=[]

    for i in range(1, len(X_clean)):
        # Training Data
        X_train=X_clean.iloc[:i]
        y_train=y_clean.iloc[:i]

        # Test Data
        X_test=X_clean.iloc[i:i+1]
        y_test=y_clean.iloc[i]
        
        model=LinearRegression()
        model.fit(X_train,y_train)
        
        #Predict Next Return
        y_pred=model.predict(X_test)[0]
        
        #Store
        predictions.append(y_pred)
        actuals.append(y_test)

    #--- Convert to Numpy Arrays ---
    predictions=np.array(predictions)
    actuals=np.array(actuals)

    #--- Evaluate R² ---
    r2=r2_score(actuals,predictions)
    print(f"Out-of-Sample R² (Linear Regression): {r2:.4f}")
    #--- Zero-Return Baseline R² ---
    zero_preds=np.zeros_like(actuals)
    r2_zero=r2_score(actuals,zero_preds)
    print(f"Out-of-Sample R² (Zero Prediction): {r2_zero:.4f}")
if __name__=="__main__":
    main()
#======================================================
