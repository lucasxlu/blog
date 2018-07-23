---
title: "[ML] Logistic Regression and Maximum Entropy Model"
mathjax: true
date: 2018-07-23 17:28:30
tags:
- Machine Learning
- Data Science
catagories:
- Algorithm
- Machine Learning
---
## Introduction
Logistic Regression是机器学习中一种非常经典且应用非常广泛的分类算法。Logistic Regression属于对数线性模型的一种。

## Logistic Regression
* Logistic Distribution: 设$X$是连续随机变量，$X$服从Logistic Distribution指的是$X$具有下列分布函数和密度函数:
  $F(x)=P(X\leq x)=\frac{1}{1+e^{-(x-\mu)/\gamma}}$

  $f(x)=\frac{e^{-(x-\mu)/\gamma}}{\gamma (1+e^{-(x-\mu)/\gamma})^2}$

二项Logistic Regression是如下的条件概率分布：  
$P(Y=1|x)=\frac{exp(w\cdot x+b)}{1+exp(w\cdot x+b)}$
$P(Y=0|x)=\frac{1}{1+exp(w\cdot x+b)}$  
对于给定的输入实例$x$，可以求得$P(Y=1|X)$和$P(Y=0|X)$，Logistic Regression比较两个条件概率值的大小，将实例$x$划分到概率值大的那一类中。

Odds: 一个事件发生的几率(odds)是指该事件发生的概率与该事件不发生的概率的比值，如果事件发生的概率是$p$，那么该事件的几率是$\frac{p}{1-p}$，该事件的对数几率(log odds)或logit函数是:   
$logit(p)=log\frac{p}{1-p}$  
对Logistic Regression而言，可得：
$log\frac{P(Y=1|x)}{1-P(Y=1|x)}=w\cdot x$

这就是说，在Logistic Regression模型中，输出$Y=1$的对数几率是输入$x$的线性函数，输出$Y=1$的对数几率是由输入$x$的线性函数表示的模型，即Logistic Regression。

### Logistic Regression参数估计
Logistic Regression学习时，可以应用 __极大似然估计法估计模型参数__，从而得到Logistic Regression Model。
设$P(Y=1|x)=\pi(x), P(Y=0|x)=1-\pi(x)$，  
似然函数为：
$\prod_{i=1}^N [\pi(x_i)]^{y_i}[1-\pi(x_i)]^{1-y_i}$
对数似然函数为  

$L(w)=\sum_{i=1}^N[y_ilog\pi(x_i)+(1-y_i)log(1-\pi(x_i))]=\sum_{i=1}^N [y_ilog\frac{\pi(x_i)}{1-\pi(x_i)}+log(1-\pi(x_i))]$

$=\sum_{i=1}^N[y_i(w\cdot x_i)-log(1+exp(w\cdot x_i))]$  

对$L(w)$求极大值，得到$w$的估计值。__Logistic Regression学习中通常采用梯度下降法和拟牛顿法__。

#### 多项式Logistic Regression
$P(Y=k|x)=\frac{exp(w_k\cdot x)}{1+\sum_{i=1}^{K-1} exp(w_k\cdot x)}, \quad k=1,2,\cdots,K-1$

$P(Y=K|x)=\frac{1}{1+\sum_{i=1}^{K-1} exp(w_k\cdot x)}$

## Maximum Entropy Model
