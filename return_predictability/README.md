# Return Predictability at Daily Frequency

## Research Question

Can daily stock returns be predicted using simple statistical or machine learning models,
or are they mostly indistinguishable from noise?

## Motivation

Many financial ML projects report high accuracy by predicting prices directly
or by using improper validation methods.
This project investigates return predictability using realistic assumptions
and time-aware validation.

## Baseline Analysis

As a first step, we evaluate whether simple historical information contains
any exploitable predictive signal at the daily frequency.

### Experimental Setup

- Target variable: daily log returns
- Features: lagged daily returns (e.g. t−1, t−5)
- Models:
  - Zero baseline (predicts zero return at all times)
  - Linear regression on lagged returns

Model performance is evaluated using in-sample R²,
which measures how much of the variance in returns can be explained by the model.

### Results

Example output from the baseline experiment:
Zero Return Baseline R^2: -0.0019
Linear Regression Baseline R^2: 0.0003

### Interpretation

Both baseline models achieve R² values close to zero.
Lagged returns explain virtually none of the variance in daily returns,
indicating an absence of meaningful linear predictive structure.

This result is expected and consistent with financial literature:
at daily frequency, asset returns behave approximately like white noise.
The purpose of this experiment is not to optimize performance,
but to establish a realistic reference point against which more complex models
must be evaluated.

Any future model must demonstrate clear improvement over these baselines
under proper time-aware validation.

## Walk-Forward Analysis — Out-of-Sample (Day 4)

In-sample evaluation can be misleading in time series settings.
To test whether any apparent signal survives under realistic conditions,
we repeat the same experiment using **expanding-window walk-forward validation**.

### Experimental Setup

- Same target and features as the baseline experiment
- **Expanding training window**
- At each time step:
  - The model is trained using only past data
  - A one-step-ahead prediction is produced
- **True out-of-sample evaluation**
- **Metric**: R² (out-of-sample)

### Results

Observed walk-forward performance:
Walk-forward OOS R^2 (Linear Regression): -0.0146
Walk-forward OOS R^2 (Zero Baseline): -0.0019


### Interpretation

Under walk-forward validation, linear regression performs **worse than the zero-return baseline**.
The negative out-of-sample R² indicates that the model fails to generalize
and that any apparent in-sample structure does not persist over time.

This result demonstrates that:

- In-sample fit does **not** imply predictability
- Lagged returns do **not** provide stable linear signal at daily frequency
- Proper time-aware validation is essential for meaningful conclusions

---

## Key Takeaways

- Daily stock returns exhibit **little to no linear predictability**
- Naive in-sample evaluation can create **illusory signals**
- Walk-forward validation reveals that simple models
  fail to outperform even trivial baselines
- The absence of signal is a **valid and informative research result**

Any future model must demonstrate **clear and robust improvement**
over these baselines under **strict out-of-sample, time-aware validation**.

---

## Next Steps

Subsequent experiments will investigate:

- Directional accuracy (sign prediction)
- Stability of predictions over time
- Why increased model complexity does not necessarily improve results

Only after establishing genuine statistical signal
will profitability-based evaluation be considered.
