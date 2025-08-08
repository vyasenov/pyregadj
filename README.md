
# pyregadj

A Python package for regression and machine learning adjustments of treatment effects in randomized experiments. 

## Installation
You can install the package using pip:

```bash
pip install pyregadj
````

## Features

* Implements difference-in-means estimator for randomized experiments
* Supports covariate-adjusted linear regression methods: pooled and groupwise
* Machine learning adjustment with support for Random Forest, Gradient Boosting, Lasso, Ridge, and Elastic Net.
* Optional cross-fitting with $K$-folds
* Clean API for pandas DataFrame input

## Quick Start

```python

import pandas as pd
import numpy as np
from pyregadj import RegAdjustRCT

# Generate dummy data
np.random.seed(1988)
n = 200

# Create covariates
age = np.random.normal(35, 10, n)
income = np.random.normal(50000, 20000, n)
education = np.random.normal(14, 3, n)

# Generate treatment assignment (50% treated)
treatment = np.random.binomial(1, 0.5, n)

# Generate outcome with treatment effect
baseline = 100 + 0.5 * age + 0.001 * income + 2 * education + np.random.normal(0, 10, n)
outcome = baseline + 15 * treatment  # True treatment effect = 15

# Create DataFrame
data = pd.DataFrame({
    'y': outcome,
    'd': treatment,
    'age': age,
    'income': income,
    'education': education
})

# Initialize calculator
te = RegAdjustRCT()

# 1. Difference-in-means
result1 = te.calculate(data, outcome='y', treatment='d', adjustment='none')

# 2. Pooled regression
result2 = te.calculate(data, outcome='y', treatment='d', adjustment='pooled', 
                      covariates=['age', 'income', 'education'])
```

## Examples

You can find detailed usage examples in the `examples/` directory.

## Statistical Methods

This package estimates average treatment effects (ATE) in randomized experiments using unadjusted, regression-adjusted, and machine learning-based estimators. All estimators assume a binary treatment variable, and they provide valid inference under standard assumptions of randomization.

### Notation

Let $Y(1)$ and $Y(0)$ denote the potential outcomes under treatment and control, respectively. The observed outcome is

$$
Y = D \cdot Y(1) + (1 - D) \cdot Y(0),
$$

where $D \in \{0,1\}$ is the binary treatment indicator. Let $X \in \mathbb{R}^p$ be a vector of covariates. We assume there are $n_1$ and $n_0$ units in the treatment ($D=1$) and control ($D=0$) groups respectively. We denote their sample standard deviations of $Y$ by $s_1$ and $s_0$. 

The estimand of interest is the average treatment effect (ATE):

$$
\tau = \mathbb{E}[Y(1) - Y(0)].
$$

### Assumptions

This package assumes that treatment assignment is randomized:

$$
(Y(1), Y(0)) \perp\!\!\!\perp D
$$

When properly implemented, randomization ensures this assumption is satisfied. Under this assumption, the estimators described below are consistent for the ATE.



### Unadjusted Estimator

This estimator compares average outcomes in the treatment and control groups:

$$
\hat{\tau}_{\text{unadj}} = \frac{1}{n_1}\sum_{i:D=1}Y - \frac{1}{n_0}\sum_{i:D=0}Y.
$$

A two-sample $t$-test with pooled variance is used to construct standard errors and confidence intervals:

$$
SE_{\text{unadj}} = \sqrt{ \hat{\sigma}_{\text{unadj}}^2 \left( \frac{1}{n_1} + \frac{1}{n_0} \right) }, 
$$

where 

$$
\hat{\sigma}_{\text{unadj}}^2 = \frac{(n_1 - 1)s_1^2 + (n_0 - 1)s_0^2}{n_1 + n_0 - 2}.
$$

This estimator is equivalent to the simple linear model

$$
Y = \alpha + \tau_{\text{unadj}} D + \varepsilon,
$$

although the regression framework allows for more flexible variance estimation.

### Pooled (Linear) Regression-Adjusted Estimator

This estimator fits a linear regression of the outcome on treatment and covariates:

$$
Y = \alpha + \tau_{\text{pool}} D + X^\top \beta + \varepsilon.
$$

The coefficient $\hat{\tau}_{\text{pool}}$ on $D$ estimates the ATE. Standard heteroskedasticity-robust standard errors (HC1) are used for inference. Optionally, covariates can be mean-centered.

This estimator can improve precision over the unadjusted estimator, particularly when covariates are predictive of the outcome.



### Groupwise (Linear) Regression-Adjusted Estimator

This estimator fits separate linear models for treatment and control groups:

$$
Y = \alpha_d + X^\top \beta_d + \varepsilon_d, \quad \text{for } D = d \in \{0,1\}.
$$

The ATE is estimated as:

$$
\hat{\tau}_{\text{grp}} = \frac{1}{n_1} \sum_{i:D=1} \hat{\mu}_1(X_i) - \frac{1}{n_0} \sum_{i:D=0} \hat{\mu}_0(X_i),
$$


where $\hat{\mu}_d(X_i)$ is the predicted outcome for group $d$ with covariate values $X_i$. 

Standard errors are computed using residual variances from the two regressions. 


$$
SE_{\text{grp}} = \sqrt{ \hat{\sigma}_{\text{grp}}^2 \left( \frac{1}{n_1} + \frac{1}{n_0} \right) },
$$

where 

$$ \hat{\sigma}_{\text{grp}}^2 = \frac{(n_1 - 1)s_{\hat{\varepsilon}_1}^2 + (n_0 - 1)s_{\hat{\varepsilon}_0}^2}{n_1 + n_0 - 2}.
$$

We now use the variance of the estimated residuals $\hat{\varepsilon}_1$, and $\hat{\varepsilon}_0$. This estimator allows covariate effects to differ by treatment group and may improve robustness and precision (similar to Lin (2013)).


### Machine Learning-Adjusted Estimators

These estimators use flexible models (e.g., random forests, gradient boosting, lasso, etc.) to estimate the conditional expectation 

$$
\mu_d(X) = \mathbb{E}[Y \mid D=d, X]  \quad \text{for } D = d \in \{0,1\}.
$$

The ATE is estimated via:

$$
\hat{\tau}_{\text{ML}} = \frac{1}{n} \sum_{i=1}^{n} \left( \hat{m}_1(X_i) - \hat{m}_0(X_i) \right)
+ \frac{D_i}{\hat{p}_1}(Y_i - \hat{m}_1(X_i))
- \frac{1 - D_i}{\hat{p}_0}(Y_i - \hat{m}_0(X_i)),
$$

where $\hat{m}_d(X_i)$ is the predicted outcome under treatment $d$, and $\hat{p}_d$ is the empirical treatment probability and $n=n_1+n_0$.

When `cross_fit=True`, predictions are obtained via sample-splitting and $K$-fold cross-fitting to mitigate overfitting. This means predictions for each unit $i$ are from models trained without $i$’s fold.

Variance is estimated from the influence functions (IFs):

$$
SE_{\text{ML}} = \sqrt{\frac{\hat{\sigma}^2_{\text{ML}}}{n}},
$$

where

$$
\hat{\sigma}^2_{\text{ML}} = \widehat{\mathrm{Var}}\!\big(IF_{1} - IF_{0}\big),
$$

and $\text{IF}_d$ is the respective IF for group $d$. 

This estimator offers a flexible and efficient approach, particularly in high dimensions when the methods above do not have attractive properties.


## Practical Considerations

* All estimators are valid under random assignment, but adjusted estimators (linear or ML) may yield narrower confidence intervals.
* Mean-centering covariates may improve interpretability but does not affect consistency.
* The ML estimators rely on user-specified models (`rf`, `gbm`, `lasso`, `ridge`, `elastic-net`) and can be cross-fit for robustness.
* For small samples, linear adjustment may be preferable due to reduced variance.
* Covariates should not be post-treatment or affected by treatment.

## References

* Lin, W. (2013). "Agnostic Notes on Regression Adjustments to Experimental Data: Reexamining Freedman’s Critique". *Annals of Applied Statistics*.
* List, J. A., Muir, I., & Sun, G. (2024). Using machine learning for efficient flexible regression adjustment in economic experiments. Econometric Reviews, 44(1), 2-40.
* Negi, A., & Wooldridge, J. M. (2021). Revisiting regression adjustment in experiments with heterogeneous treatment effects. Econometric Reviews, 40(5), 504-534.
* Wu, E., & Gagnon-Bartsch, J. A. (2018). The LOOP estimator: Adjusting for covariates in randomized experiments. Evaluation review, 42(4), 458-488.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

To cite this package in publications, please use the following BibTeX entry:

```bibtex
@misc{yasenov2025pytreateffects,
  author       = {Vasco Yasenov},
  title        = {pytreateffects: Treatment Effect Estimation for Randomized Experiments in Python},
  year         = {2025},
  howpublished = {\url{https://github.com/vyasenov/pytreateffects}},
  note         = {Version 0.1.0}
}
```