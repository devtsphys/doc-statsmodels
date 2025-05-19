# doc-statsmodels

# Statsmodels Reference Card & Cheat Sheet

## Table of Contents
1. [Introduction to statsmodels](#introduction-to-statsmodels)
2. [Installation & Setup](#installation--setup)
3. [Linear Regression Models](#linear-regression-models)
4. [Time Series Analysis](#time-series-analysis)
5. [Panel Data Models](#panel-data-models)
6. [Discrete Choice Models](#discrete-choice-models)
7. [Nonparametric Models](#nonparametric-models)
8. [Statistical Tests](#statistical-tests)
9. [GLM (Generalized Linear Models)](#glm-generalized-linear-models)
10. [Diagnostic Tests & Model Evaluation](#diagnostic-tests--model-evaluation)
11. [Forecasting](#forecasting)
12. [Best Practices](#best-practices)

## Introduction to statsmodels

statsmodels is a Python package that provides classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests and exploring statistical data.

**Key Features:**
- Regression models
- Time series analysis
- Discrete choice models
- Statistical tests
- Plotting functionality
- Statistical computation framework
- Comprehensive results analysis

## Installation & Setup

```python
# Installing statsmodels
pip install statsmodels

# Basic imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
```

## Linear Regression Models

### Ordinary Least Squares (OLS)

```python
# Method 1: Using the formula API
model = smf.ols('y ~ x1 + x2 + x3', data=df)
results = model.fit()

# Method 2: Using arrays (add constant manually)
X = sm.add_constant(X)  # Adds a column of 1s for the intercept
model = sm.OLS(y, X)
results = model.fit()

# Accessing results
print(results.summary())
print(results.params)  # Coefficients
print(results.bse)     # Standard errors
print(results.rsquared, results.rsquared_adj)  # R-squared values
print(results.aic, results.bic)  # Information criteria
print(results.pvalues)  # p-values
print(results.conf_int())  # Confidence intervals
```

### Weighted Least Squares (WLS)

```python
# For heteroskedastic data
weights = 1 / (standard_deviations ** 2)
model = sm.WLS(y, X, weights=weights)
results = model.fit()
```

### Robust Linear Regression

```python
# Handle outliers with robust regression
from statsmodels.robust.robust_linear_model import RLM
model = RLM(y, X)
results = model.fit()
```

### Quantile Regression

```python
from statsmodels.regression.quantile_regression import QuantReg
model = QuantReg(y, X)
results = model.fit(q=0.5)  # Median regression (q=0.5)
```

## Time Series Analysis

### ARIMA Models

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA(p,d,q) model
model = ARIMA(y, order=(p, d, q))  # p=AR order, d=difference, q=MA order
results = model.fit()

# Forecasting
forecast = results.forecast(steps=5)  # Forecast next 5 periods
```

### SARIMAX Models (Seasonal ARIMA with exogenous variables)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA with seasonal component and exogenous variables
model = SARIMAX(y, 
                exog=X,
                order=(p, d, q), 
                seasonal_order=(P, D, Q, s))  # s = seasonal period
results = model.fit()

# Forecast with exogenous variables
forecast = results.forecast(steps=5, exog=X_future)
```

### Vector Autoregression (VAR)

```python
from statsmodels.tsa.vector_ar.var_model import VAR

# Fit VAR model
model = VAR(data)
results = model.fit(maxlags=5)  # Auto-selects lags based on information criteria

# Granger causality tests
results.test_causality('y1', ['y2'], kind='f')
```

### Dynamic Factor Models

```python
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

model = DynamicFactor(data, k_factors=2, factor_order=1)
results = model.fit()
```

### Stationarity Tests

```python
from statsmodels.tsa.stattools import adfuller, kpss

# Augmented Dickey-Fuller test (null: unit root exists)
adf_result = adfuller(y)
print(f'ADF p-value: {adf_result[1]}')

# KPSS test (null: series is stationary)
kpss_result = kpss(y)
print(f'KPSS p-value: {kpss_result[1]}')
```

### Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series into trend, seasonal, and residual components
decomposition = seasonal_decompose(y, model='additive', period=12)
decomposition.plot()
```

## Panel Data Models

### Fixed Effects Models

```python
# Using formula API with entity and/or time fixed effects
model = smf.ols('y ~ x1 + x2 + EntityEffects', data=panel_data)
results = model.fit()

# Alternative using PanelOLS from linearmodels package
from linearmodels.panel import PanelOLS
model = PanelOLS(y, X, entity_effects=True)
results = model.fit()
```

### Random Effects Models

```python
from statsmodels.regression.mixed_linear_model import MixedLM

# Random effects model
model = MixedLM(y, X, groups=panel_data['group'])
results = model.fit()
```

## Discrete Choice Models

### Logistic Regression (Logit)

```python
# For binary outcome variables
model = sm.Logit(y, X)
results = model.fit()

# Using formula API
model = smf.logit('outcome ~ x1 + x2', data=df)
results = model.fit()

# Predict probabilities
predicted_probs = results.predict(X_new)
```

### Probit Model

```python
model = sm.Probit(y, X)
results = model.fit()
```

### Multinomial Logit (MNLogit)

```python
# For categorical outcomes with more than two categories
model = sm.MNLogit(y, X)
results = model.fit()

# Predict probabilities for each category
predicted_probs = results.predict(X_new)
```

### Ordered Models

```python
from statsmodels.miscmodels.ordinal_model import OrderedModel

# For ordinal categorical variables
model = OrderedModel(y, X, distr='logit')  # or 'probit'
results = model.fit()
```

## Nonparametric Models

### Kernel Regression

```python
from statsmodels.nonparametric.kernel_regression import KernelReg

# Fit kernel regression
kr = KernelReg(y, X, var_type='c' * X.shape[1])  # 'c' for continuous
mean, mfx = kr.fit(X_new)
```

### Kernel Density Estimation

```python
from statsmodels.nonparametric.kde import KDEUnivariate, KDEMultivariate

# Univariate KDE
kde = KDEUnivariate(data)
kde.fit()
kde.plot()

# Multivariate KDE
kde = KDEMultivariate(data, var_type='cc')  # 'c' for continuous variables
density = kde.pdf(points_to_evaluate)
```

## Statistical Tests

### Normality Tests

```python
from statsmodels.stats.diagnostic import lilliefors
from scipy import stats

# Shapiro-Wilk test
w, p_value = stats.shapiro(data)

# Jarque-Bera test
jb, p_value = stats.jarque_bera(data)

# Kolmogorov-Smirnov test (Lilliefors variant)
ks, p_value = lilliefors(data)
```

### Autocorrelation Tests

```python
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

# Ljung-Box test
lb, p_value = acorr_ljungbox(residuals, lags=[10])

# Durbin-Watson test
dw = durbin_watson(residuals)
```

### Heteroskedasticity Tests

```python
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

# Breusch-Pagan test
bp_test = het_breuschpagan(residuals, X)
print(f'BP p-value: {bp_test[1]}')

# White test
white_test = het_white(residuals, X)
print(f'White p-value: {white_test[1]}')
```

### Engle's ARCH Test

```python
from statsmodels.stats.diagnostic import het_arch

# Test for ARCH effects in residuals
arch_test = het_arch(residuals)
print(f'ARCH p-value: {arch_test[1]}')
```

## GLM (Generalized Linear Models)

### Common GLM Models

```python
# Poisson regression (for count data)
model = sm.GLM(y, X, family=sm.families.Poisson())
results = model.fit()

# Negative Binomial (for overdispersed count data)
model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
results = model.fit()

# Gamma regression (for positive continuous data)
model = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.log()))
results = model.fit()
```

### Zero-Inflated Models

```python
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP

# Zero-inflated Poisson model
model = ZeroInflatedPoisson(y, X)
results = model.fit()

# Zero-inflated Negative Binomial model
model = ZeroInflatedNegativeBinomialP(y, X)
results = model.fit()
```

## Diagnostic Tests & Model Evaluation

### Multicollinearity Detection

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)  # VIF > 10 suggests multicollinearity issues
```

### Outlier Detection and Influence

```python
from statsmodels.stats.outliers_influence import OLSInfluence

# Influence measures for OLS
influence = OLSInfluence(results)

# Cook's distance
cooks_d = influence.cooks_distance[0]

# DFBETAS
dfbetas = influence.dfbetas

# Plot leverage vs. normalized residuals squared
fig, ax = plt.subplots(figsize=(8, 6))
influence.plot_influence(ax=ax, criterion="cooks")
```

### Model Comparison

```python
# Compare nested models with likelihood ratio test
from statsmodels.stats.anova import anova_lm

# Full model vs. restricted model
anova_results = anova_lm(results_restricted, results_full)
print(anova_results)

# Compare non-nested models with information criteria
print(f"Model 1 AIC: {results1.aic}, BIC: {results1.bic}")
print(f"Model 2 AIC: {results2.aic}, BIC: {results2.bic}")
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

# K-fold cross-validation with statsmodels + scikit-learn
def ols_predict(X, train_idx, test_idx):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test = X[test_idx]
    
    model = sm.OLS(y_train, sm.add_constant(X_train))
    results = model.fit()
    
    return results.predict(sm.add_constant(X_test))

# Then use with cross_val_score
```

## Forecasting

### Time Series Forecasting Workflow

```python
# 1. Split data into train and test
train = data[:split_point]
test = data[split_point:]

# 2. Fit model on training data
model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
results = model.fit()

# 3. Forecast
forecast = results.get_forecast(steps=len(test))
forecast_ci = forecast.conf_int()

# 4. Plot results
plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Actual Test Data')
plt.plot(test.index, forecast.predicted_mean, label='Forecast')
plt.fill_between(test.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], 
                 color='k', alpha=0.1, label='95% Confidence Interval')
plt.legend()
plt.title('Time Series Forecast')
```

### Forecast Evaluation

```python
from statsmodels.tools.eval_measures import rmse, mse, mae

# Calculate error metrics
y_true = test.values
y_pred = forecast.predicted_mean.values

print(f'RMSE: {rmse(y_true, y_pred)}')
print(f'MSE: {mse(y_true, y_pred)}')
print(f'MAE: {mae(y_true, y_pred)}')

# Mean Absolute Percentage Error (MAPE)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f'MAPE: {mape(y_true, y_pred)}%')
```

## Best Practices

### Data Preparation

1. **Handle missing values appropriately**:
   ```python
   # Check for missing values
   df.isna().sum()
   
   # Options: Drop, fill with mean/median, or use more advanced imputation
   df = df.dropna()  # Or
   df = df.fillna(df.mean())
   ```

2. **Scale features when necessary**:
   ```python
   from sklearn.preprocessing import StandardScaler
   
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

3. **Check for multicollinearity**:
   - Calculate correlation matrix
   - Use VIF scores (see diagnostic section)

### Model Selection

1. **Start simple**:
   - Begin with a simple model and add complexity as needed
   - Use information criteria (AIC, BIC) to guide model selection

2. **Cross-validate when possible**:
   - Use time-series cross-validation for time-series data
   - Use k-fold for non-time dependent data

3. **Compare nested models with F-tests or likelihood ratio tests**:
   ```python
   from statsmodels.stats.anova import anova_lm
   
   anova_results = anova_lm(results_restricted, results_full)
   ```

### Results Interpretation

1. **Focus on effect sizes, not just p-values**:
   ```python
   # Calculate confidence intervals
   conf_int = results.conf_int()
   
   # For easier interpretation of coefficients in log models
   np.exp(results.params)  # Exponentiate for interpretation as multiplicative effects
   ```

2. **Validate assumptions**:
   - Linearity: Check residual plots
   - Independence: Check for autocorrelation
   - Normality: Q-Q plots, normality tests
   - Homoscedasticity: Residual plots, statistical tests

3. **Use robust standard errors when necessary**:
   ```python
   # For heteroskedasticity
   results = model.fit(cov_type='HC3')
   
   # For clustered data
   results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_variable})
   ```

### Common Pitfalls to Avoid

1. **Using p-values alone for variable selection**:
   - Consider domain knowledge and effect sizes
   - Use regularization approaches for high-dimensional data

2. **Ignoring model assumptions**:
   - Always check model assumptions with diagnostics
   - Use appropriate models for different data types

3. **Overfitting**:
   - Use cross-validation
   - Consider regularization or simpler models

4. **Not addressing endogeneity**:
   - Use instrumental variables or panel methods when appropriate
   - Consider causal inference methods for causality questions

# Python statsmodels Reference Card

## Table of Contents
- [Installation and Import](#installation-and-import)
- [Data Preparation](#data-preparation)
- [Linear Regression Models](#linear-regression-models)
- [Discrete Choice Models](#discrete-choice-models)
- [Time Series Analysis](#time-series-analysis)
- [ANOVA and ANCOVA](#anova-and-ancova)
- [Nonparametric Methods](#nonparametric-methods)
- [Generalized Linear Models (GLM)](#generalized-linear-models-glm)
- [Robust Linear Models](#robust-linear-models)
- [Mixed Linear Models](#mixed-linear-models)
- [Multivariate Models](#multivariate-models)
- [Hypothesis Testing](#hypothesis-testing)
- [Diagnostics and Specification Tests](#diagnostics-and-specification-tests)
- [Plotting and Visualization](#plotting-and-visualization)

## Installation and Import

```python
# Installation
pip install statsmodels

# Common imports
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
```

## Data Preparation

| Function | Description | Example |
|----------|-------------|---------|
| `sm.add_constant()` | Add a constant (intercept) to a model | `X = sm.add_constant(X)` |
| `sm.datasets` | Access built-in datasets | `data = sm.datasets.get_rdataset("mtcars").data` |
| `sm.tools.tools.categorical()` | Convert categorical variables to dummy variables | `dummies = sm.tools.tools.categorical(x, drop=True)` |
| `sm.tsa.filters.hp_filter()` | Hodrick-Prescott filter for time series | `cycle, trend = sm.tsa.filters.hp_filter(data, lamb=1600)` |

## Linear Regression Models

| Model/Function | Description | Example |
|----------------|-------------|---------|
| `sm.OLS()` | Ordinary Least Squares | ```model = sm.OLS(y, X).fit()``` |
| `smf.ols()` | Formula-based OLS | ```model = smf.ols('y ~ x1 + x2', data=df).fit()``` |
| `model.summary()` | Model summary | ```print(model.summary())``` |
| `model.params` | Model parameters | ```coefficients = model.params``` |
| `model.predict()` | Make predictions | ```predictions = model.predict(X_new)``` |
| `model.conf_int()` | Confidence intervals | ```ci = model.conf_int(alpha=0.05)``` |
| `model.rsquared` | R-squared value | ```rsq = model.rsquared``` |
| `model.rsquared_adj` | Adjusted R-squared | ```adj_rsq = model.rsquared_adj``` |
| `model.aic` | Akaike Information Criterion | ```aic = model.aic``` |
| `model.bic` | Bayesian Information Criterion | ```bic = model.bic``` |
| `model.pvalues` | p-values | ```pvals = model.pvalues``` |
| `model.tvalues` | t-statistics | ```tvals = model.tvalues``` |
| `model.fvalue` | F-statistic | ```fval = model.fvalue``` |
| `model.mse_model` | Mean squared error of model | ```mse = model.mse_model``` |
| `model.mse_resid` | Mean squared error of residuals | ```mse_r = model.mse_resid``` |
| `model.bse` | Standard errors of parameters | ```std_errors = model.bse``` |
| `sm.WLS()` | Weighted Least Squares | ```model = sm.WLS(y, X, weights=w).fit()``` |
| `sm.GLS()` | Generalized Least Squares | ```model = sm.GLS(y, X, sigma=sigma).fit()``` |
| `sm.GLSAR()` | ARMA errors regression | ```model = sm.GLSAR(y, X, rho=1).fit()``` |
| `sm.RegressionResults.compare_f_test()` | F-test for nested models | ```model1.compare_f_test(model2)``` |
| `sm.OLS.from_formula()` | Alternative formula API | ```model = sm.OLS.from_formula('y ~ x1 + x2', data=df).fit()``` |

## Discrete Choice Models

| Model/Function | Description | Example |
|----------------|-------------|---------|
| `sm.Logit()` | Logistic regression | ```model = sm.Logit(y, X).fit()``` |
| `smf.logit()` | Formula-based logistic regression | ```model = smf.logit('y ~ x1 + x2', data=df).fit()``` |
| `sm.Probit()` | Probit regression | ```model = sm.Probit(y, X).fit()``` |
| `smf.probit()` | Formula-based probit regression | ```model = smf.probit('y ~ x1 + x2', data=df).fit()``` |
| `sm.MNLogit()` | Multinomial logistic regression | ```model = sm.MNLogit(y, X).fit()``` |
| `smf.mnlogit()` | Formula-based multinomial logistic | ```model = smf.mnlogit('y ~ x1 + x2', data=df).fit()``` |
| `model.predict()` | Predict probabilities | ```probs = model.predict(X_new)``` |
| `model.pred_table()` | Prediction table | ```table = model.pred_table()``` |
| `model.summary()` | Model summary | ```print(model.summary())``` |
| `model.pvalues` | p-values | ```pvals = model.pvalues``` |
| `model.params` | Model parameters | ```params = model.params``` |
| `model.llf` | Log-likelihood | ```ll = model.llf``` |
| `model.llnull` | Null log-likelihood | ```ll0 = model.llnull``` |
| `model.pseudo_rsquared` | McFadden's pseudo R-squared | ```pseudo_r2 = model.prsquared``` |

## Time Series Analysis

| Model/Function | Description | Example |
|----------------|-------------|---------|
| `sm.tsa.ARMA()` | ARMA model | ```model = sm.tsa.ARMA(y, order=(p,q)).fit()``` |
| `sm.tsa.ARIMA()` | ARIMA model | ```model = sm.tsa.ARIMA(y, order=(p,d,q)).fit()``` |
| `sm.tsa.statespace.SARIMAX()` | SARIMAX model | ```model = sm.tsa.statespace.SARIMAX(y, order=(p,d,q), seasonal_order=(P,D,Q,s)).fit()``` |
| `sm.tsa.VAR()` | Vector Autoregression | ```model = sm.tsa.VAR(y).fit(maxlags=5)``` |
| `sm.tsa.VECM()` | Vector Error Correction Model | ```model = sm.tsa.VECM(y, k_ar_diff=2, coint_rank=1).fit()``` |
| `sm.tsa.statespace.ExponentialSmoothing()` | Exponential smoothing | ```model = sm.tsa.statespace.ExponentialSmoothing(y).fit()``` |
| `sm.tsa.statespace.UnobservedComponents()` | Unobserved components model | ```model = sm.tsa.statespace.UnobservedComponents(y, level='local linear trend', seasonal=12).fit()``` |
| `sm.tsa.statespace.DynamicFactor()` | Dynamic factor model | ```model = sm.tsa.statespace.DynamicFactor(y, k_factors=1, factor_order=1).fit()``` |
| `sm.tsa.seasonal_decompose()` | Seasonal decomposition | ```result = sm.tsa.seasonal_decompose(y, model='additive')``` |
| `sm.tsa.stattools.adfuller()` | Augmented Dickey-Fuller test | ```result = sm.tsa.stattools.adfuller(y)``` |
| `sm.tsa.stattools.kpss()` | KPSS test for stationarity | ```result = sm.tsa.stattools.kpss(y)``` |
| `sm.tsa.stattools.acf()` | Autocorrelation function | ```acf = sm.tsa.stattools.acf(y)``` |
| `sm.tsa.stattools.pacf()` | Partial autocorrelation function | ```pacf = sm.tsa.stattools.pacf(y)``` |
| `sm.tsa.stattools.grangercausalitytests()` | Granger causality test | ```granger_test = sm.tsa.stattools.grangercausalitytests(data, maxlag=2)``` |
| `sm.tsa.stattools.arma_order_select_ic()` | ARMA order selection | ```best_order = sm.tsa.stattools.arma_order_select_ic(y, max_ar=5, max_ma=5)``` |
| `model.forecast()` | Forecast future values | ```forecasts = model.forecast(steps=5)``` |
| `model.get_forecast()` | Forecast with confidence intervals | ```forecast = model.get_forecast(steps=5)``` |
| `model.impulse_responses()` | Impulse response functions (VAR) | ```irf = model.impulse_responses(steps=10)``` |
| `model.plot_diagnostics()` | Diagnostic plots | ```model.plot_diagnostics()``` |

## ANOVA and ANCOVA

| Model/Function | Description | Example |
|----------------|-------------|---------|
| `sm.stats.anova_lm()` | ANOVA for linear models | ```anova_table = sm.stats.anova_lm(model)``` |
| `smf.ols('y ~ C(group)', data=df).fit()` | One-way ANOVA | ```model = smf.ols('y ~ C(group)', data=df).fit()``` |
| `smf.ols('y ~ C(group1) + C(group2)', data=df).fit()` | Two-way ANOVA | ```model = smf.ols('y ~ C(group1) + C(group2)', data=df).fit()``` |
| `smf.ols('y ~ C(group1) * C(group2)', data=df).fit()` | Two-way ANOVA with interaction | ```model = smf.ols('y ~ C(group1) * C(group2)', data=df).fit()``` |
| `smf.ols('y ~ C(group) + x', data=df).fit()` | ANCOVA | ```model = smf.ols('y ~ C(group) + x', data=df).fit()``` |
| `smf.ols('y ~ C(group) * x', data=df).fit()` | ANCOVA with interaction | ```model = smf.ols('y ~ C(group) * x', data=df).fit()``` |

## Nonparametric Methods

| Model/Function | Description | Example |
|----------------|-------------|---------|
| `sm.nonparametric.KDEUnivariate()` | Kernel Density Estimation | ```kde = sm.nonparametric.KDEUnivariate(data).fit()``` |
| `sm.nonparametric.KDEMultivariate()` | Multivariate Kernel Density Estimation | ```kde = sm.nonparametric.KDEMultivariate(data=data, var_type='c' * dim).fit()``` |
| `sm.nonparametric.lowess()` | LOWESS smoothing | ```smoothed = sm.nonparametric.lowess(y, x, frac=0.3)``` |
| `sm.nonparametric.KernelReg()` | Kernel regression | ```kreg = sm.nonparametric.KernelReg(y, X, var_type='c' * X.shape[1])``` |

## Generalized Linear Models (GLM)

| Model/Function | Description | Example |
|----------------|-------------|---------|
| `sm.GLM()` | Generalized Linear Model | ```model = sm.GLM(y, X, family=sm.families.Gaussian()).fit()``` |
| `smf.glm()` | Formula-based GLM | ```model = smf.glm('y ~ x1 + x2', data=df, family=sm.families.Gaussian()).fit()``` |
| `sm.families.Gaussian()` | Gaussian family (identity link) | ```family = sm.families.Gaussian()``` |
| `sm.families.Binomial()` | Binomial family (logit link) | ```family = sm.families.Binomial()``` |
| `sm.families.Poisson()` | Poisson family (log link) | ```family = sm.families.Poisson()``` |
| `sm.families.NegativeBinomial()` | Negative binomial family | ```family = sm.families.NegativeBinomial(alpha=1.0)``` |
| `sm.families.Gamma()` | Gamma family (inverse link) | ```family = sm.families.Gamma()``` |
| `sm.families.InverseGaussian()` | Inverse Gaussian family | ```family = sm.families.InverseGaussian()``` |
| `sm.families.Tweedie()` | Tweedie family | ```family = sm.families.Tweedie(var_power=1.5)``` |
| `family.link` | Link function | ```link = family.link``` |
| `model.predict()` | Predict values | ```predictions = model.predict(X_new)``` |
| `model.summary()` | Model summary | ```print(model.summary())``` |
| `model.deviance` | Model deviance | ```dev = model.deviance``` |
| `model.null_deviance` | Null deviance | ```null_dev = model.null_deviance``` |
| `model.pearson_chi2` | Pearson chi-squared | ```chi2 = model.pearson_chi2``` |

## Robust Linear Models

| Model/Function | Description | Example |
|----------------|-------------|---------|
| `sm.RLM()` | Robust Linear Model | ```model = sm.RLM(y, X).fit()``` |
| `smf.rlm()` | Formula-based Robust Linear Model | ```model = smf.rlm('y ~ x1 + x2', data=df).fit()``` |
| `sm.robust.scale.mad()` | Median Absolute Deviation | ```mad = sm.robust.scale.mad(data)``` |
| `sm.robust.scale.huber()` | Huber's scaling | ```huber_scale = sm.robust.scale.huber(data)``` |
| `model.weights` | Robust weights | ```weights = model.weights``` |

## Mixed Linear Models

| Model/Function | Description | Example |
|----------------|-------------|---------|
| `sm.MixedLM()` | Mixed Linear Model | ```model = sm.MixedLM(y, X, groups=groups).fit()``` |
| `smf.mixedlm()` | Formula-based Mixed LM | ```model = smf.mixedlm('y ~ x1', data=df, groups=df['group']).fit()``` |
| `model.random_effects` | Random effects | ```re = model.random_effects``` |
| `model.fixed_effects` | Fixed effects | ```fe = model.fixed_effects``` |

## Multivariate Models

| Model/Function | Description | Example |
|----------------|-------------|---------|
| `sm.multivariate.MANOVA()` | Multivariate Analysis of Variance | ```mv = sm.multivariate.MANOVA(y, x)``` |
| `sm.multivariate.PCA()` | Principal Component Analysis | ```pca = sm.multivariate.PCA(data, standardize=True).fit()``` |
| `sm.multivariate.factor.Factor()` | Factor Analysis | ```fa = sm.multivariate.factor.Factor(data, n_factor=2).fit()``` |

## Hypothesis Testing

| Model/Function | Description | Example |
|----------------|-------------|---------|
| `sm.stats.ttest_ind()` | Independent samples t-test | ```result = sm.stats.ttest_ind(group1, group2)``` |
| `sm.stats.ztest()` | Z-test | ```result = sm.stats.ztest(data, value=0)``` |
| `sm.stats.weightstats.DescrStatsW()` | Weighted statistics | ```stats = sm.stats.weightstats.DescrStatsW(data, weights=weights)``` |
| `stats.ttest_mean()` | One-sample t-test | ```t_test = stats.ttest_mean(3)``` |
| `sm.stats.proportion.proportions_ztest()` | Test for proportions | ```z_score, p_value = sm.stats.proportion.proportions_ztest(count, nobs)``` |
| `sm.stats.diagnostic.lilliefors()` | Lilliefors test for normality | ```test_stat, p_value = sm.stats.diagnostic.lilliefors(data)``` |
| `sm.stats.power.tt_ind_solve_power()` | Power analysis for t-test | ```sample_size = sm.stats.power.tt_ind_solve_power(effect_size=0.5, alpha=0.05, power=0.8)``` |

## Diagnostics and Specification Tests

| Model/Function | Description | Example |
|----------------|-------------|---------|
| `sm.stats.diagnostic.het_breuschpagan()` | Breusch-Pagan test for heteroskedasticity | ```lm, lm_pvalue, fvalue, f_pvalue = sm.stats.diagnostic.het_breuschpagan(residuals, X)``` |
| `sm.stats.diagnostic.het_white()` | White's test for heteroskedasticity | ```lm, lm_pvalue, fvalue, f_pvalue = sm.stats.diagnostic.het_white(residuals, X)``` |
| `sm.stats.diagnostic.acorr_ljungbox()` | Ljung-Box test for autocorrelation | ```lb, p = sm.stats.diagnostic.acorr_ljungbox(residuals)``` |
| `sm.stats.diagnostic.acorr_breusch_godfrey()` | Breusch-Godfrey test for autocorrelation | ```lm, lm_pvalue, fvalue, f_pvalue = sm.stats.diagnostic.acorr_breusch_godfrey(results)``` |
| `sm.stats.diagnostic.linear_reset()` | RESET test for functional form | ```reset = sm.stats.diagnostic.linear_reset(results)``` |
| `sm.stats.diagnostic.linear_harvey_collier()` | Harvey-Collier test for linearity | ```hc = sm.stats.diagnostic.linear_harvey_collier(results)``` |
| `sm.stats.diagnostic.linear_rainbow()` | Rainbow test for linearity | ```rainbow = sm.stats.diagnostic.linear_rainbow(results)``` |
| `sm.stats.diagnostic.variance_inflation_factor()` | Variance Inflation Factor | ```vif = sm.stats.diagnostic.variance_inflation_factor(X, i)``` |
| `sm.stats.outliers_influence.OLSInfluence()` | Influence and outlier measures | ```influence = sm.stats.outliers_influence.OLSInfluence(results)``` |
| `influence.summary_frame()` | Summary of influence measures | ```inf_sum = influence.summary_frame()``` |
| `influence.cooks_distance` | Cook's distance | ```cooks_d = influence.cooks_distance``` |
| `influence.hat_matrix_diag` | Leverage (hat matrix diagonal) | ```leverage = influence.hat_matrix_diag``` |
| `influence.resid_studentized` | Studentized residuals | ```stud_resid = influence.resid_studentized``` |

## Plotting and Visualization

| Model/Function | Description | Example |
|----------------|-------------|---------|
| `sm.graphics.plot_fit()` | Plot fitted vs. actual values | ```sm.graphics.plot_fit(results, 0)``` |
| `sm.graphics.plot_regress_exog()` | Component-component plus residual plot | ```sm.graphics.plot_regress_exog(results, 'x1')``` |
| `sm.graphics.plot_partregress()` | Partial regression plot | ```sm.graphics.plot_partregress('y', 'x1', ['x2', 'x3'], data=df)``` |
| `sm.graphics.plot_ccpr()` | Component-component plus residual plot | ```sm.graphics.plot_ccpr(results, 'x1')``` |
| `sm.graphics.plot_partregress_grid()` | Grid of partial regression plots | ```sm.graphics.plot_partregress_grid(results)``` |
| `sm.graphics.plot_leverage_resid2()` | Influence plot | ```sm.graphics.plot_leverage_resid2(results)``` |
| `sm.graphics.qqplot()` | Quantile-quantile plot | ```sm.graphics.qqplot(residuals)``` |
| `sm.graphics.influence_plot()` | Influence plot | ```sm.graphics.influence_plot(results)``` |
| `sm.graphics.tsa.plot_acf()` | Autocorrelation plot | ```sm.graphics.tsa.plot_acf(series)``` |
| `sm.graphics.tsa.plot_pacf()` | Partial autocorrelation plot | ```sm.graphics.tsa.plot_pacf(series)``` |
| `model.plot_diagnostics()` | Model diagnostic plots | ```model.plot_diagnostics(figsize=(12, 8))``` |
