---
title: "[ML] Model Selection and Performance Metric"
catalog: false
mathjax: true
date: 2018-07-19 11:02:28
tags:
- Machine Learning
- Data Science
catagories:
- Algorithm
- Machine Learning
---
## Feature Selection
* 当$p>n$时，无法使用Backward Selection，但Forward Selection可以。

* $Precision=\frac{TP}{TP+FP}$  
  $Recall=\frac{TP}{TP+FN}$  
  $F_1=\frac{2PR}{P+R}$  

* 很多时候我们有多个二分类confusionmatrix，例如进行多次training/test，每次得到一个confusion matrix，或是在多个数据集上进行training/test，希望估计算法的全局性能。总之，我们希望在$n$ 个二分类confusion matrix上综合考查Precision和Recall：
    * Macro-P/Macro-R/Macro-F1: 先在各个confusion matrix上分别计算 P/R/F1，再计算均值：
    $Macro-P=\frac{1}{n}\sum_{i=1}^n P_i$  
    $Macro-R=\frac{1}{n}\sum_{i=1}^n R_i$  
    $Macro-F_1=\frac{2\times Macro-P \times Macro-R}{Macro-P+Macro-R}$
    
    * Micro-P/Micro-R/Micro-F1: 先将各confusion matrix元素平均，得到TP, FP, TN, FN的均值，再计算：
    $Micro-P=\frac{\bar{TP}}{\bar{TP}+\bar{FP}}$  
    $Micro-R=\frac{\bar{TP}}{\bar{TP}+\bar{FN}}$  
    $Micro-F_1=\frac{2\times Micro-P\times Micro-R}{Micro-P+Micro-R}$

* ROC曲线的纵轴是"真正率"(TPR)，横轴是"假正率"(FPR)：
    $TPR=\frac{TP}{TP+FN}$  
    $FPR=\frac{FP}{FP+TN}$
现实任务中，通常是利用有限个test set samples来绘制ROC，此时仅能获得有限个(TPR, FPR)坐标对，只能绘制比较粗糙(带锯齿)的ROC曲线。

* 泛化误差可分解为: $bias^2+variance+\epsilon^2$
  $bias$度量了learner本身的拟合能力；
  $variance$度量了同样大小training set变动所导致的性能变化，即刻画了数据扰动所造成的影响；
  $\epsilon$表达了当前任务上任何learner所能达到的期望泛化误差下界，即刻画了学习问题本身的难度。