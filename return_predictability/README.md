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
