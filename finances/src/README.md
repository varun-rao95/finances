# Simulating spend
## Overview


We wish to model daily spend data that can appear in multiple “buckets.” Each bucket $b$ has:

A Bernoulli indicator $Y_b∈{0,1}$, denoting whether bucket $b$ is active on a given day.
A positive spend distribution (e.g. lognormal, gamma) conditional on $Y_b=1$.

We incorporate dependence across the $b$ buckets via a Gaussian copula on the indicator variables ${Y_b}$.

### 1. Fit Marginal Models

For each bucket $b$, we fit:

A logistic (or Bernoulli) model giving
$p_b=Pr⁡(Y_b=1)$

A lognormal (or gamma) regression for the positive spend amount $X_b$​ conditioned on $Y_b=1$.

e.g a lognormal model with intercept $β0_b$​ and scale $σ_b$​, the conditional PDF is
$$
f_{Xb∣Yb=1}(x)  =  Lognormal ⁣(β_{0,b}, σ_b^2)
$$.

Hence bucket $b$ has the marginal “two‐part” distribution:

$$
Pr⁡(spend=0)=1−p_b \text{ , and if  } spend>0 ,X_b∼Lognormal(β_{0,b},σ_b^2)
$$

### 2. Construct Indicators & Estimate Correlation

For each day $t$, define the indicator $Y_b(t)=1$ if bucket $b$ is observed (i.e. nonzero) on day $t$. Otherwise 0.

Compute the correlation $Σ$ among the columns ${Yb}$.

### 3. Build the Copula Simulation

We simulate $Z∈R^B$ as follows:

Draw $Z∼N(0, Σ)$ from a multivariate normal with mean $0$ and covariance $Σ$.

Convert each component to a uniform:
$$
U_b  =  Φ(Z_b),
$$
where $Φ$ is the standard normal CDF.

### 4. Threshold for Bernoulli Indicators

For each bucket $b$, we set:
$$
Y_b^{(\mathrm{sim})} =
\begin{cases}
1, & \text{if } U_b \le p_b, \\
0, & \text{otherwise}.
\end{cases}
$$

This ensures:
$$
Pr⁡(Y_b^{(sim)})  =  p_b,
$$

matching the marginal probability from the fitted logistic model.

### 5. Draw Positive Amounts Conditionally

If $Y_b^{(sim)}=1$, we draw a positive spend from bucket $b$’s fitted distribution. For a lognormal with intercept $β_{0,b}$​ and scale $σ_b$​:
$$
X_b^{(sim)}  ∼  exp⁡(β_{0,b}+σ_b Z^∗)\text{ for some random } Z^∗ ∼ N(0,1).
$$

If $Y_b^{(sim)}=0$ , set $X_b^{(sim)}=0$ .