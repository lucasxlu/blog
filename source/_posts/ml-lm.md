---
title: "[ML] Linear Model"
catalog: false
mathjax: true
date: 2018-07-18 12:23:24
tags:
- Machine Learning
- Data Science
catagories:
- Algorithm
- Machine Learning
---
## Statistical Methods
* 参数方法，例如Linear Model。参数方法容易拟合，且可解释性强，但也存在问题。以Linear Model为例，若真实存在的模型本身就不是Linear，那就会有问题。
* 非参数方法：不需要显式指定$f$的形式，例如KNN。非参数方法的优点在于：因没有显示指定$f$，故Non-Parameter Methods可以拟合比Parameter Methods更广的函数。

## Linear Model
* Standard error can be used to compute confidence intervals. A 95% confidence interval is defined as a range of values such that with 95% probability, the range will contain the true unknown value of the parameter. The range is defined in terms of lower and upper limits computed from the sample of data. For linear regression, the 95% confidence interval for $\beta1$ approximately takes the form:

    $\hat{\beta}_1\pm 2\cdot SE(\hat{\beta}_1)$

    That is, there is approximately a 95% chance that the interval $[\hat{\beta}_1- 2\cdot SE(\hat{\beta}_1), \hat{\beta}_1+ 2\cdot SE(\hat{\beta}_1)]$ will contain the true value of $\beta_1$.

* Standard error也可用于假设检验：  
  原假设$H_0$: $X$和$Y$之间没有关系，即$\beta_1= 0$  
  备择假设$H_{\alpha}$: $X$和$Y$之间有关系，即$\beta_1\neq 0$  
  
  因此我们需要判断$\hat{\beta}_1$ (即$\beta$的估计值)是否远离0。这取决于$SE(\hat{\beta}_1)$，若$SE(\hat{\beta}_1)$非常小，那么即便是相对较小的$\hat{\beta}_1$值也可以足够肯定$\beta_1\neq 0$。若$SE(\hat{\beta}_1)$非常大，那么$\hat{\beta}_1$的绝对值必须足够大才能拒绝原假设。

* RSS measures the amount of variability that is left unexplained after performing the regression.

* TSS-RSS measures the amount of variability in the response that is explained by performing the regression. 

* $R^2$ measures the proportion of variability in $Y$ that can be explained using $X$.

* 可以使用 __residual plot__ 来分析Outliers，离得太远的可以被视为是outliers，但是如何量化"多远"这个度量呢？可以使用 __studentized residual__，计算方式为$e_i/SE(e_i)$。

  __Observations whose studentized residuals are greater than 3 in absolute value are possible outliers.__

* A simple way to detect collinearity is to look at the correlation matrix of the predictors. An element of this matrix is large in absolute value indicates a pair of highly correlated variables, and therefore a collinearity problem in the data.

  However, it is possible for collinearity to exist between three or more variables even if no pair of variable has a particularly high correlation. We call this __''multi-collinearity''__.

  A better way to detect multi-collinearity is to use VIF (variance inflation factor). VIF is the ratio of the variance of $\hat{\beta}_j$ when fitting the full model divided by the variance of $\hat{\beta}_j$ if fits on its own.

  $$
  VIF(\hat{\beta_j})=\frac{1}{1-R_{X_j|X_{-j}}^2}
  $$
  
  VIF 最小为1，代表完全没有共线性，VIF大于5或10时代表有比较严重的共线性。

* In high dimensions there is effectively a reduction in sample size. High dimensions result in a phenomenon in which a given observation has no nearby neighbors ——that is called __curse of dimensionality__. That is, the K observations that are nearest to a given test observation $x_0$ may be very far away from $x_0$ in p-dimensional space when p is large, leading to a very poor prediction of $f(x_0)$ and hence a pour KNN fit.

* Generally, Parameter-Methods will tend to outperform non-parameter methods when there's a small number of observations per predictors.