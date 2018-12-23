---
title: "[ML] Model Selection and Performance Metric"
catalog: false
mathjax: true
date: 2018-12-23 16:53:28
tags:
- Machine Learning
- Data Science
catagories:
- Algorithm
- Machine Learning
---
## Introduction
* **Overfitting的本质**：过拟合(overfitting)是指在模型参数拟合过程中的问题，由于训练数据包含“抽样误差”，训练时，复杂的模型将抽样误差也考虑在内，将抽样误差也进行了很好的拟合。

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

* Cosine Similarity VS Euclidean Distance  
  cosine similarity关注的是```向量之间的角度关系，并不关心它们的绝对大小。```因此cosine similarity在高维情况下依然可以保持```相同时为1，正交时为0，相反时为-1```的性质，不受feature vector的维度影响；但是Euclidean Distance的数值则受到维度的影响，范围不固定。**若feature vector经过了normalization，此时Euclidean Distance与Cosine Similarity有着单调的关系**。
  $$
  ||A-B||_2=\sqrt{2(1-cos(A, B))}
  $$
  在此场景下，使用cosine similarity与euclidean distance的结果是相同的。

* 超参数调优
    * **Grid Search**：查找搜索范围内的所有点来确定最优值，若采用较大的搜索范围和较小的步长，Grid Search有很大概率找到全局最优解。但是该方法非常耗时，在实际应用中，Grid Search一般会先使用较广的搜索范围和较大的步长，来寻找全局最优解的可能位置；然后逐步缩小搜索范围和步长，来寻找更加精确的值。
    * **Random Search**：idea和Grid Search比较相似，只是不再测试上界和下界之间的所有值，而是在搜索范围中随机选取样本点。
    * **Bayesian Optimization**：Grid Search和Random Search在测试一个新点时，会忽略前一个点的信息；而Bayesian Optimization则充分利用了之前点的信息。它通过对object function的形状进行学习，找到使object function向全局最优值提升的参数。即首先根据先验分布，假设一个搜集函数。然后每一次使用新的采样点来测试object function时，利用这个信息来更新object function的先验分布。最后，算法测试由后验分布给出的全局最优值可能出现的位置的点。需要注意的是，Bayesian Optimization一旦找到了一个局部最优值，它会在该区域不断采样，所以很容易陷入局部最优值。所以需要在搜索和利用之间找到一个良好的tradeoff。