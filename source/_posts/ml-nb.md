---
title: "[ML] Naive Bayes"
catalog: false
mathjax: true
date: 2018-07-19 16:17:20
tags:
- Machine Learning
- Data Science
catagories:
- Algorithm
- Machine Learning
---
## Introduction
1. Naive Bayes 是基于Bayes Theorem与 __特征条件独立__ 假设的分类算法。对于给定的数据集，首先基于特征条件独立假设 __学习输入/输出的联合概率分布__；然后基于此模型，对给定的输入$x$，利用Bayes Theorem求出后验概率最大的输出$y$。

2. Naive Bayes通过训练数据集学习联合概率分布$P(X,Y)$。Naive Bayes对条件概率分布做了条件独立性假设：
$$
P(X=x|Y=c_k)=P(X^{(1)}=x^{(1)},\cdots,X^{(n)}=x^{(n)}|Y=c_k)
$$

Naive Bayes实际上学习到生成数据的机制，所以属于 __生成模型__。条件独立性假设等于是说 __用于分类的特征在类确定的情况下都是条件独立的__。

## 公式推导
$$
P(Y=c_k|X=x)=\frac{P(X=x|Y=c_k)P(Y=c_k)}{\sum_{k}P(X=x|Y=c_k)P(Y=c_k)}=
\frac{P(Y=c_k)\prod_j P(X^{(j)}=x^{(j)}|Y=c_k)}{\sum_k P(Y=c_k)\prod_j P(X^{(j)}=x^{(j)}|Y=c_k)}    
$$
因上式中分母对所有$c_k$都是相同的，所以：
$$
y=\mathop{argmax}\limits_{c_k}P(Y=c_k)\prod_j P(X^{(j)}=x^{(j)}|Y=c_k)
$$

采用0-1损失函数：
$$
L(Y,f(X))=
\begin{cases}
    1, & Y\neq f(X)\\
    0, & otherwise
\end{cases}
$$
条件期望为：
$$
R_{exp}(f)=E_x \sum_{k=1}^K[L(c_k,f(X))]P(c_k|X)
$$
因此：
$$
f(x)=\mathop{argmin}\limits_{y\in \mathcal{Y}} \sum_{k=1}^KL(c_k,y)P(c_k|X=x)\\
=\mathop{argmin}\limits_{y\in \mathcal{Y}} \sum_{k=1}^K P(y\neq c_k|X=x)=\mathop{argmin}\limits_{y\in \mathcal{Y}} (1-P(y=c_k|X=x))\\
=\mathop{argmax}\limits_{y\in \mathcal{Y}} P(y=c_k|X=x)
$$
这样一来，根据期望风险最小化就得到了后验概率最大化准则：
$$
f(x)=\mathop{argmax}\limits_{c_k}P(c_k|X=x)
$$

先验概率$P(Y=c_k)$的极大似然估计是：
$$
P(Y=c_k)=\frac{\sum_{i=1}^NI(y_i=c_k)}{N}, k=1,2,\cdots,K
$$
设第$j$个特征$x^{(j)}$可能取值的集合为$\{a_{j1},a_{j2},\cdots,a_{jS_j}\}$，条件概率$P(X^{(j)}=a_{jl}|Y=c_k)$的极大似然估计是：
$$
P(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum_{i=1}^NI(x_i^{(j)}=a_{jl},y_i=c_k)}{\sum_{i=1}^N I(y_i=c_k)}
$$
