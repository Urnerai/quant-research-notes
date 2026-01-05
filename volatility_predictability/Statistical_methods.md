# Quant Research Checkpoint  
## Volatility Forecasting — From Naive Benchmark Onward

This document continues the quant research pipeline starting from the **naive volatility benchmark**.  
The focus is **not return direction**, but whether **risk (variance / volatility)** is predictable in a disciplined, out-of-sample framework.

The structure follows:
**intuition → mathematics → practical interpretation**.

---

## 1. Notation and Setup

- \( P_t \): asset price at time \( t \)
- \( r_t \): daily return
- Volatility is **not directly observable**
- Daily volatility must be approximated using **proxies**
- Objective:  
  At time \( t \), produce a forecast for \( t+1 \):
  \[
  \widehat{RV}_{t+1}
  \]

---

## 2. Daily Returns

### Log return definition
\[
r_t = \log\left(\frac{P_t}{P_{t-1}}\right)
\]

**Intuition**
- Measures relative (percentage) price change
- Log returns are approximately additive over time

**Practical note**
For small changes:
\[
\log(1+x) \approx x
\]

This makes log returns convenient for modeling and aggregation.

---

## 3. Why Volatility Needs a Proxy

Volatility is a property of a **distribution**, not a single observation:
\[
\text{Volatility} \sim \sqrt{\mathbb{E}[r_t^2]}
\]

On a daily frequency, we only observe **one return**, so true volatility is unobservable.

Therefore, we use **proxies**.

---

## 4. Volatility Proxies

### 4.1 Realized Variance (RV)

\[
RV_t = r_t^2
\]

**Intuition**
- Large absolute returns imply high risk
- Squaring removes the sign and emphasizes extreme moves

**Practical interpretation**
- Common in academic literature
- Sensitive to outliers (which is often acceptable in risk modeling)

---

### 4.2 Absolute Return (AV)

\[
AV_t = |r_t|
\]

**Intuition**
- Less aggressive than squaring
- More robust to extreme observations

**Note**
Both proxies are used in practice; this project primarily uses \( RV_t \).

---

## 5. Naive Volatility Benchmark (Random Walk Variance)

### Definition
\[
\widehat{RV}_{t+1} = RV_t
\]

**Intuition**
> Tomorrow’s risk equals today’s risk.

**Role in research**
- This is the **null hypothesis**: no predictability
- Any reasonable model must outperform this baseline out-of-sample

If a model cannot beat the naive benchmark, it has **no research value**.

---

## 6. Stronger Baselines (Benchmark Set)

Relying on a single naive benchmark is insufficient.  
A proper volatility study uses multiple baselines.

---

### 6.1 Rolling Mean (SMA)

\[
\widehat{RV}_{t+1} = \frac{1}{m} \sum_{i=0}^{m-1} RV_{t-i}
\]

**Intuition**
- Average recent risk over a fixed window

**Window choices**
- \( m = 5 \): short-term
- \( m = 20 \): monthly
- \( m = 60 \): quarterly

**Interpretation**
Volatility is **persistent**, so smoothing often improves forecasts.

---

### 6.2 EWMA (Exponentially Weighted Moving Average)

\[
\widehat{RV}_{t+1}
= \lambda \widehat{RV}_t + (1 - \lambda) RV_t
\]

**Intuition**
- Recent observations matter more
- Older information decays exponentially but never disappears

**Expanded form**
\[
\widehat{RV}_{t+1}
= (1-\lambda)\sum_{k=0}^{\infty} \lambda^k RV_{t-k}
\]

**Parameter meaning**
- \( \lambda \rightarrow 1 \): long memory, smooth series
- \( \lambda \rightarrow 0 \): short memory, fast reaction

**Practical note**
- RiskMetrics suggests \( \lambda \approx 0.94 \) for daily data
- Forecasts are always positive by construction

---

### 6.3 Expanding Mean (Long-Run Average)

\[
\widehat{RV}_{t+1}
= \frac{1}{t} \sum_{i=1}^{t} RV_i
\]

**Intuition**
- Long-term average risk level

**Interpretation**
- Slow to adapt to regime changes
- Useful as a structural reference

---

## 7. Evaluation Metrics

Volatility forecasts should **never** be evaluated using a single metric.

---

### 7.1 Mean Squared Error (MSE)

\[
\text{MSE}
= \frac{1}{T} \sum_{t} (RV_t - \widehat{RV}_t)^2
\]

**Pros**
- Simple and intuitive

**Cons**
- Dominated by extreme observations
- Can be misleading for heavy-tailed data

---

### 7.2 QLIKE (Quasi-Likelihood Loss)

\[
\text{QLIKE}
= \frac{1}{T} \sum_t
\left(
\log(\widehat{RV}_t)
+ \frac{RV_t}{\widehat{RV}_t}
\right)
\]

**Key intuition**
- Underestimating volatility is heavily penalized
- Overestimation is also penalized, but asymmetrically

**Why this matters**
In risk management, **underestimating risk is worse than overestimating it**.

**Implementation note**
Forecasts must be strictly positive:
\[
\widehat{RV}_t \leftarrow \max(\widehat{RV}_t, \varepsilon)
\]

---

## 8. Walk-Forward (Out-of-Sample) Framework

Avoiding data leakage is critical.

**Rule**
- At time \( t \), use only data \( \leq t \)
- Forecast \( t+1 \)

### Expanding window approach
- Train: \( 1 \ldots t \)
- Predict: \( t+1 \)
- Move forward one step

**Alignment checklist**
- Features computed using data up to \( t \)
- Target strictly at \( t+1 \)
- Forecast and realized series correctly aligned

---

## 9. First Model Beyond Baselines: HAR

HAR captures volatility persistence across **multiple time scales**.

---

### 9.1 HAR Features

Daily:
\[
RV_t^{(d)} = RV_t
\]

Weekly (5 days):
\[
RV_t^{(w)} = \frac{1}{5} \sum_{i=0}^{4} RV_{t-i}
\]

Monthly (22 days):
\[
RV_t^{(m)} = \frac{1}{22} \sum_{i=0}^{21} RV_{t-i}
\]

---

### 9.2 HAR Regression

\[
RV_{t+1}
= \beta_0
+ \beta_d RV_t^{(d)}
+ \beta_w RV_t^{(w)}
+ \beta_m RV_t^{(m)}
+ \epsilon_{t+1}
\]

**Intuition**
Markets contain heterogeneous agents:
- short-term traders
- medium-term portfolio managers
- long-term risk allocators

HAR models this heterogeneity explicitly.

---

### 9.3 Log-HAR (Preferred in Practice)

\[
\log(RV_{t+1})
= \beta_0
+ \beta_d \log(RV_t^{(d)})
+ \beta_w \log(RV_t^{(w)})
+ \beta_m \log(RV_t^{(m)})
+ \epsilon
\]

**Why log-transform**
- Ensures positivity
- Reduces outlier impact
- Improves numerical stability

---

## 10. GARCH(1,1) — Brief Reference

\[
r_t = \sigma_t \epsilon_t, \quad \epsilon_t \sim (0,1)
\]
\[
\sigma_t^2
= \omega
+ \alpha r_{t-1}^2
+ \beta \sigma_{t-1}^2
\]

**Interpretation**
- Recent shocks increase risk
- Volatility is self-persistent

**Stability condition**
\[
\alpha + \beta < 1
\]

---

## 11. Recommended Handwritten Practice

For each item, write:
- the formula
- one sentence of intuition

1. \( RV_t = r_t^2 \)  
2. Naive: \( \widehat{RV}_{t+1} = RV_t \)  
3. EWMA formula and role of \( \lambda \)  
4. QLIKE and why underestimation is costly  
5. HAR daily / weekly / monthly features  

---

## 12. Next Coding Checkpoint

Implementation should be **incremental**, not monolithic:

1. Return calculation and \( RV_t \)
2. Walk-forward loop (naive only)
3. Baselines: SMA, EWMA, expanding mean
4. Evaluation: MSE and QLIKE
5. HAR features and regression
6. (Optional) GARCH with rolling refits

---

**Rule of thumb**  
If a model cannot beat EWMA + QLIKE out-of-sample,  
it is not a production-quality volatility model.
