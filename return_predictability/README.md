# Return Predictability at Daily Frequency

## Research Question

Can daily stock returns be predicted using simple statistical or machine learning models,
or are they largely indistinguishable from noise under realistic assumptions?

---

## Motivation

Many financial machine learning projects report strong predictive performance
by modeling prices directly or by relying on improper validation schemes.
Such approaches often introduce look-ahead bias or evaluate models
under unrealistic assumptions.

This project investigates daily **return predictability**
under **strict time-aware validation**, with the explicit goal of avoiding
illusory in-sample signals.

The objective is not to optimize performance,
but to determine whether any **statistically meaningful signal**
exists at the daily frequency.

---

## Baseline Analysis (In-Sample)

As an initial step, we evaluate whether simple historical information
contains any exploitable predictive structure when assessed in-sample.

### Experimental Setup

- Target variable: daily log returns
- Features: lagged daily returns (e.g. t−1, t−5)
- Models:
  - Zero baseline (predicts zero return at all times)
  - Linear regression on lagged returns
- Metric: in-sample R²

### Results

Example output from the baseline experiment:

- Zero Return Baseline R²: −0.0019  
- Linear Regression Baseline R²: 0.0003  

### Interpretation

Both baseline models achieve R² values close to zero.
Lagged returns explain virtually none of the variance in daily returns,
indicating an absence of meaningful linear predictive structure.

While this result is expected and consistent with financial literature,
in-sample evaluation alone is insufficient to draw conclusions
about predictability in time series settings.

---

## Walk-Forward Analysis (Out-of-Sample)

To test whether any apparent in-sample structure survives
under realistic conditions, we repeat the same experiment
using **expanding-window walk-forward validation**.

### Experimental Setup

- Same target and features as the baseline experiment
- Expanding training window
- At each time step:
  - The model is trained using only past data
  - A one-step-ahead prediction is produced
- True out-of-sample evaluation
- Metric: out-of-sample R²

### Results

Observed walk-forward performance:

| Model / Configuration                | OOS R²   |
|-------------------------------------|----------|
| Linear Regression (Short Lags)       | −0.0146  |
| Linear Regression (Extended Lags)    | −0.8615  |
| Linear Regression (Permuted Targets) | −0.0072  |
| Ridge Regression (Extended Lags)     | −0.0045  |

### Interpretation

Across all configurations, out-of-sample R² remains **non-positive**.

Key observations:

- **Baseline failure**  
  Linear regression with short lagged returns fails to outperform
  the zero-return baseline (OOS R² = −0.0146).

- **Lag horizon robustness**  
  Extending the lag set substantially worsens performance
  (OOS R² = −0.8615), indicating noise amplification rather than
  recovery of predictive signal.

- **Permutation sanity check**  
  Performance with permuted targets is of similar magnitude
  (OOS R² = −0.0072), suggesting that observed performance
  is indistinguishable from noise.

- **Regularization**  
  Ridge regression slightly stabilizes the model but does not recover
  positive out-of-sample performance
  (OOS R² = −0.0045).

---

## Robustness & Closure

Additional robustness checks were conducted to verify that the observed
null result is not an artifact of modeling or validation choices.

Across extended lag specifications, regularization,
and permutation-based sanity checks,
out-of-sample performance remains consistently non-positive.

These results indicate that the absence of daily return predictability
is **structural**, rather than a consequence of specific modeling assumptions.

---

## Key Takeaways

- Daily stock returns exhibit **little to no stable linear predictability**
- Naive in-sample evaluation can create **illusory signals**
- Proper walk-forward validation reveals that simple models
  fail to outperform even trivial baselines
- A null result, when robustly established, is a **valid and informative**
  research outcome

---

## Next Steps

Given the absence of return predictability at the daily frequency,
subsequent analysis will shift focus from returns themselves
to **return structure**, including:

- Volatility and absolute return dynamics
- Regime-dependent behavior
- Conditional predictability under different market states

This transition forms the basis of **Project 02**.
