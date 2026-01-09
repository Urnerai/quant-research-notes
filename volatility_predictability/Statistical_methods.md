# Quant Research Checkpoint  
## Volatility Forecasting — From Naive Benchmark Onward

This document extends the quantitative research pipeline starting from the **naive volatility benchmark**.  
The objective is **not return direction**, but to evaluate whether **risk (variance / volatility)** is predictable in a disciplined **out-of-sample** framework.

The exposition follows:  
**intuition → mathematics → practical interpretation**

---

## 1. Notation and Setup

- P(t): asset price at time t  
- r(t): daily log return  
- Volatility is **not directly observable**  
- Daily volatility must be approximated using **observable proxies**

**Goal**  
At time t, produce a volatility forecast for t+1.

---

## 2. Daily Returns

**Log return definition**

```
r(t) = log( P(t) / P(t-1) )
```

**Intuition**
- Measures relative price change
- Approximately additive over time

**Practical note**

For small x:

```
log(1 + x) ≈ x
```

This simplifies aggregation and modeling.

---

## 3. Why Volatility Needs a Proxy

Volatility is a property of a **distribution**, not a single observation.

Conceptually:

```
volatility ~ sqrt( E[ r(t)^2 ] )
```

At daily frequency we observe only **one return**, so true volatility is unobservable.

Therefore, we rely on **volatility proxies**.

---

## 4. Volatility Proxies

### 4.1 Realized Variance (RV)

```
RV(t) = r(t)^2
```

**Intuition**
- Large absolute returns imply higher risk
- Squaring removes sign and emphasizes extremes

**Practical interpretation**
- Standard proxy in volatility literature
- Outlier-sensitive (acceptable for risk modeling)

---

### 4.2 Absolute Return (AV)

```
AV(t) = | r(t) |
```

**Intuition**
- Less aggressive than squaring
- More robust to extreme observations

**Note**  
This project primarily uses RV(t).

---

## 5. Naive Volatility Benchmark (Random Walk Variance)

**Definition**

```
RV(t+1) = RV(t)
```

**Intuition**
> Tomorrow’s risk equals today’s risk.

**Role in research**
- Serves as the **null hypothesis**
- Any model must beat this out-of-sample

A model that cannot beat this benchmark has **no empirical value**.

---

## 6. Stronger Baselines

A single naive benchmark is insufficient.  
Robust volatility research requires **multiple baselines**.

---

### 6.1 Rolling Mean (Simple Moving Average)

```
RV(t+1) = (1/m) * sum_{i=0 to m-1} RV(t-i)
```

**Typical windows**
- m = 5   → short-term
- m = 20  → monthly
- m = 60  → quarterly

**Interpretation**
Volatility is persistent; smoothing often improves stability.

---

### 6.2 EWMA (Exponentially Weighted Moving Average)

```
RV(t+1) = λ * RV(t) + (1 - λ) * RV(t)
```

Expanded form:

```
RV(t+1) = (1 - λ) * sum_{k=0 to ∞} λ^k * RV(t-k)
```

**Intuition**
- Recent observations matter more
- Older information decays smoothly

**Parameter meaning**
- λ → 1 : long memory, smooth forecasts
- λ → 0 : short memory, fast reaction

**Practical note**
- RiskMetrics suggests λ ≈ 0.94 for daily data
- Forecasts are always positive

---

### 6.3 Expanding Mean (Long-Run Average)

```
RV(t+1) = (1/t) * sum_{i=1 to t} RV(i)
```

**Interpretation**
- Captures long-term average risk
- Slow to adapt to regime changes

---

## 7. Evaluation Metrics

Volatility forecasts must be evaluated using **more than one metric**.

---

### 7.1 Mean Squared Error (MSE)

```
MSE = (1/T) * sum_t ( RV(t) - RV_hat(t) )^2
```

**Pros**
- Simple and intuitive

**Cons**
- Dominated by extreme observations

---

### 7.2 QLIKE (Quasi-Likelihood Loss)

```
QLIKE = (1/T) * sum_t [ log(RV_hat(t)) + RV(t) / RV_hat(t) ]
```

**Key intuition**
- Underestimating volatility is penalized heavily
- Overestimation is penalized asymmetrically

**Why this matters**
In risk management, **underestimating risk is worse than overestimating it**.

**Implementation note**

```
RV_hat(t) = max( RV_hat(t), ε )
```

---

## 8. Walk-Forward (Out-of-Sample) Framework

**Rule**
- At time t, use only data available up to t
- Forecast t+1

**Expanding window**
- Train on 1 → t
- Predict t+1
- Move forward one step

**Alignment checklist**
- Features use data ≤ t
- Target is strictly t+1
- Forecast and realized values are aligned

---

## 9. First Model Beyond Baselines: HAR

HAR captures volatility persistence across **multiple time scales**.

---

### 9.1 HAR Features

Daily:

```
RV_d(t) = RV(t)
```

Weekly (5 days):

```
RV_w(t) = (1/5) * sum_{i=0 to 4} RV(t-i)
```

Monthly (22 days):

```
RV_m(t) = (1/22) * sum_{i=0 to 21} RV(t-i)
```

---

### 9.2 HAR Regression

```
RV(t+1) =
  β0
+ βd * RV_d(t)
+ βw * RV_w(t)
+ βm * RV_m(t)
+ ε(t+1)
```

---

### 9.3 Log-HAR (Preferred)

```
log( RV(t+1) ) =
  β0
+ βd * log( RV_d(t) )
+ βw * log( RV_w(t) )
+ βm * log( RV_m(t) )
+ ε
```

---

## 10. GARCH(1,1) — Brief Reference

```
r(t) = σ(t) * ε(t)
σ(t)^2 = ω + α * r(t-1)^2 + β * σ(t-1)^2
```

**Stability condition**

```
α + β < 1
```

---

## 11. Handwritten Practice

1. RV(t) = r(t)^2  
2. Naive: RV(t+1) = RV(t)  
3. EWMA and λ  
4. QLIKE and underestimation risk  
5. HAR daily / weekly / monthly  

---

## 12. Next Coding Checkpoint

1. Returns and RV(t)
2. Walk-forward loop (naive)
3. Baselines: SMA, EWMA, expanding mean
4. Evaluation: MSE, QLIKE
5. HAR regression
6. (Optional) Rolling GARCH

---

**Rule of thumb**  
If a model cannot beat **EWMA under QLIKE** out-of-sample,  
it is **not** a production-quality volatility model.
