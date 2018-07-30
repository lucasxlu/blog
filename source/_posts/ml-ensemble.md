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

Boosting主要关注降低bias，因此Boosting能基于泛化性能相当弱的weak learner构建出很强的集成。而bagging主要降低variance，因此它在不剪枝决策树、NN等易受样本扰动的learner上效用更为明显。

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

#### AdaBoost算法的解释
AdaBoost还可以认为是模型为加法模型、Loss Function为指数函数、学习算法为前向分步算法时的二分类学习方法。

##### 前向分步算法
考虑加法模型：  
$f(x)=\sum_{m=1}^M \beta_m b(x;\gamma_m)$  
其中，$b(x;\gamma_m)$为基函数的参数，$\beta_m$为基函数的系数。

在给定training set及Loss Function $L(y,f(x))$的条件下，学习加法模型$f(x)$成为经验风险极小化即Loss Function极小化问题：  
$\mathop{min} \limits_{\beta_m, \gamma_m} \sum_{i=1}^N L\big(y_i,\sum_{m=1}^M \beta_m b(x_i;\gamma_m)\big)$

前向分步算法求解复杂优化问题的思想是：因为学习的是加法模型，如果能够从前往后，每一步只学习一个基函数及其系数，逐步逼近优化目标函数式，那么就可以简化优化的复杂度。具体的，每一步只需要优化以下Loss Function：  
$\mathop{min} \limits_{\beta, \gamma}\sum_{i=1}^N L(y_i,\beta b(x_i;\gamma))$

1. 初始化$f_0(x)=0$
2. 对$m=1,2,\cdots,M$
    * 极小化Loss：  
      $(\beta_m, \gamma_m)=\mathop{argmin} \limits_{\beta, \gamma} \sum_{i=1}^N L\big(y_i,f_{m-1}(x_i)+\beta b(x_i;\gamma)\big)$  
      得到参数$\beta_m, \gamma_m$

    * 更新  
      $f_m(x)=f_{m-1}(x)+\beta_m b(x;\gamma_m)$
3. 得到加法模型  
   $f(x)=f_M(x)=\sum_{m=1}^M \beta_m b(x;\gamma_m)$

#### Boosting Tree
Boosting方法实际采用加法模型(即基函数的线性组合)与前向分步算法，以决策树为基函数的Boosting方法称为Boosting Tree。Boosting Tree模型可以表示为决策树的加法模型：  
$f_M(x)=\sum_{m=1}^M T(x;\Theta_m)$  
其中$T(x;\Theta_m)$表示决策树，$\Theta_m$表示决策树参数，$T$为树的个数。

Boosting Tree采用前向分步算法，首先确定初始提升树$f_0(x)=0$，第$m$步的模型是：  
$f_m(x)=f_{m-1}(x)+T(x;\Theta_m)$

其中，$f_{m-1}(x)$为当前模型，通过经验风险最小化确定下一棵决策树的参数$\Theta$，  
$\hat{\Theta}_m=\mathop{argmin} \limits_{\Theta_m} \sum_{i=1}^N L(y_i,f_{m-1}(x_i) + T(x_i;\Theta_m))$

若将输入空间$\chi$划分为$J$个互不相交的区域$R_1,R_2,\cdots,R_J$，并且在每个区域上确定输出的常量$c_j$，那么树可以表示为：  
$T(x;\Theta)=\sum_{j=1}^Jc_j I(x\in R_j)$  
其中，参数$\Theta=\{(R_1,c_1), (R_2,c_2), \cdots, (R_J,c_J)\}$表示树的区域划分和各区域上的常数，$J$是回归树的复杂度即叶结点个数。

回归问题Boosting Tree使用以下前向分步算法：  
$$
f_0(x)=0  \\
f_m(x)=f_{m-1}(x)+T(x;\Theta_m) \\
f_M(x)=\sum_{m=1}^M T(x;\Theta_m)
$$

在前向分步算法的第$m$步，给定当前模型$f_{m-1}(x)$，需求解：  
$\hat{\Theta}_m=\mathop{argmin} \limits_{\Theta_m} \sum_{i=1}^N L(y_i,f_{m-1}(x_i) + T(x_i;\Theta_m))$  
得到$\hat{\Theta}_m$，即第$m$棵树的参数。

采用MSE Loss时，  
$L(y,f(x))=(y-f(x))^2$

其损失变为：  
$L(y,f_{m-1}(x) + T(x;\Theta_m))=[y-f_{m-1}(x)-T(x;\Theta_m)]^2=[r-T(x;\Theta_m)]^2$

这里，$r=y-f_{m-1}(x)$是模型拟合数据的残差。所以， __对回归问题的Boosting Tree来说，只需简单地拟合当前模型的残差__。

##### Boosting Tree for Regression
1. 初始化$f_0(x)=0$
2. 对$m=1,2,\cdots,M$
   * 计算残差：$r_{mi}=y_i-f_{m-1}(x_i)$
   * 拟合残差学习一个回归树，得到$T(x;\Theta_m)$
   * 更新$f_m(x)=f_{m-1}(x)+T(x;\Theta_m)$
3. 得到回归问题Boosting Tree：  
   $f_M(x)=\sum_{m=1}^M T(x;\Theta_m)$

#### Gradient Boosting (GBDT)
Boosting Tree利用加法模型与前向分步算法实现学习的优化过程，当Loss Function是MSE和指数Loss时，每一步的优化是很简单的。但对于一般的Loss Function而言，往往每一步优化并不容易，这一问题可以利用Gradient Boosting解决。这是利用Gradient Descend的近似方法，其关键是利用Loss Function的负梯度在当前模型的值：  
$$-[ \frac{\partial L(y,f(x_i))}{\partial f(x_i)} ]_{f(x)=f_{m-1}(x)}$$
作为回归问题提升树算法中的残差近似值，拟合一个回归树。

1. 初始化$f_0(x)=\mathop{argmin} \limits_{c} \sum_{i=1}^N L(y_i,c)$
2. 对$m=1,2,\cdots,M$
   * 对$i=1,2,\cdots,N$，计算：  
     $r_{mi}=-[ \frac{\partial L(y,f(x_i))}{\partial f(x_i)} ]_{f(x)=f_{m-1}(x)}$
   * 对$r_{mi}$拟合一个回归树，得到第$m$棵树的叶结点区域$R_{mj},\quad j=1,2, \cdots,J$
   * 对$j=1,2,\cdots,J$计算：  
     $c_{mj}=\mathop{argmin} \limits_{c} \sum_{x_i\in R_{mj}} L(y_i,f_{m-1}(x_i)+c)$
   * 更新$f_m(x)=f_{m-1}(x)+\sum_{j=1}^J c_{mj}I(x\in R_{mj})$
3. 得到回归树：  
   $$\hat{f}(x)=f_M(x)=\sum_{m=1}^M \sum_{j=1}^J c_{mj} I(x\in R_{mj})$$