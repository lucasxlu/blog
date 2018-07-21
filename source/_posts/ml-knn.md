---
title: "[ML] KNN"
catalog: false
mathjax: true
date: 2018-07-19 12:37:17
subtitle:
header-img: "Demo.png"
tags:
- Machine Learning
- Data Science
catagories:
- Algorithm
- Machine Learning
---
## 介绍
KNN是机器学习中一种非常常见的分类/回归算法。在 __分类__ 时，根据majority voting来选择相应的预测Label；在 __回归__ 时，可取K个距离最近点的mean作为预测值。K值的选择、Distance Metric、分类决策规则是KNN算法的关键。

### 常用的 Distance Metric
$L_p(x_i, x_j)=(\sum_{l=1}^n |x_i^{(l)}-x_j^{(l)}|^p)^{\frac{1}{p}}$
当$p=\infty$时，它是各个坐标的极大值：$L_{\infty}(x_i, x_j)=\mathop{max}\limits_{l}|x_i^{(l)}-x_j^{(l)}|$

### K值的选择
* 当K取较小值时，模型比较复杂，variance比较大，容易发生过拟合；当K取较大值时，模型比较简单，bias比较大，容易发生欠拟合。