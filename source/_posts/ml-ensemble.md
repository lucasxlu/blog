---
title: "[ML] Ensemble Learning"
date: 2018-07-25 10:59:08
mathjax: true
tags:
- Machine Learning
- Data Science
catagories:
- Algorithm
- Machine Learning
---
## Introduction
Ensemble Learning是ML中一个非常热门的领域，也是很多比赛Top方案的必选。本文对常见的Ensemble Learning做一个简要介绍。

根据Base Learner的生成方式，目前的Ensemble Learning方法大致可以分为两大类：即base learner之间存在强依赖关系、必须串行生成的序列化方法，以及base learner间不存在强依赖关系、可同时生成的并行化方法；前者的代表是Boosting，后者的代表是Bagging和Random Forests。

## Ensemble Learning Algorithm
### Boosting
Boosting是一种常用的统计学习方法，应用广泛且有效，在分类问题中，它通过改变训练样本的权重，学习多个分类器，并将这些分类器进行线性组合，提高分类性能。

Boosting方法就是从weak learner出发，反复学习，得到一系列weak learner(base learner)，然后组合这些weak learner，构成一个强分类器。大多数的提升方法都是改变training set的概率分布，针对不同的training set分布调用weak learner algorithm学习一系列weak learner。

AdaBoost的做法是，先从初始training set中训练一个base learner，再根据base learner的表现对训练样本分布进行调整，使得先前base learner做错的样本在后续受到更多的关注，然后基于调整后的样本分布来训练下一个base learner；如此重复进行，直至base learner数目达到事先指定的值$T$，最终将这$T$个base learner进行加权结合。AdaBoost采用weighted majority voting的做法，具体的，加大分类误差率较小的weak learner的权重，使其在表决中起较大作用，减小分类误差大的weak learner的权重，使其在表决中起较小的作用。

#### AdaBoost
1. 初始化training set的权值分布
$D_1=(w_{11},\cdots,w_{1i},\cdots,w_{1N}),w_{1i}=\frac{1}{N}$
2. 对$m=1,2,\cdots,M$
    * 使用具有权值分布$D_m$的training set学习，得到base learner：  
$G_m(x):\chi \to \{-1,+1\}$
    * 计算$G_m(x)$在training set上的分类误差率：  
$e_m=P(G_m(x_i)\neq y_i)=\sum_{i=1}^N w_{mi}I(G_m(x_i)\neq y_i)$  
$w_{mi}$是第$m$轮中第$i$个实例的权值。
    * 计算$G_m(x)$的系数：  
$\alpha_m=\frac{1}{2}log \frac{1-e_m}{e_m}$
    * 更新training set的权值分布：  
$D_{m+1}=(w_{m+1,1},\cdots,w_{m+1,i},\cdots,w_{m+1,N})$  
$w_{m+1,i}=\frac{w_{mi}}{Z_m}exp(-\alpha_m y_i G_m(x_i))$  
    这里，$Z_m$是规范化因子：  
$Z_m=\sum_{i=1}^Nw_{mi}exp(-\alpha_m y_i G_m(x_i))$  
它使得$D_{mi}$成为一个概率分布。
    * 构建base learner的线性组合：  
$f(x)=\sum_{m=1}^M \alpha_m G_m(x)$  
得到最终分类器：  
$f(x)=sign(f(x))=sign\big(\sum_{m=1}^M \alpha_m G_m(x)\big)$  
注：$\alpha_m$之和并不为1。




