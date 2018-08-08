---
title: "[DL] Optimization Algorithm in Deep Learning"
catalog: false
date: 2018-07-20 11:46:37
mathjax: true
tags:
- Machine Learning
- Deep Learning
- Optimization
- Data Science
catagories:
- Algorithm
- Machine Learning
- Deep Learning
- Optimization
---
## 介绍
与很多传统机器学习算法相比，由于深度神经网络的本身特性(例如非凸、高度非线性等)，使得整个目标函数的优化极为困难。因此优化算法是深度学习领域一个非常非常重要的组成部分。私以为，Deep Learning主要有3大组件：1) 网络结构，2) Loss Function, 3) 优化算法。虽然目前Paper中大多数都是设计Networks Architecture + Loss Function，然后SGD/Adam Optimizer一波带走，但笔者还是觉得有必要把这些优化算法来一个自己的整理与总结的。

> 注：本文大多数内容来自花书《[Deep Learning](https://www.deeplearningbook.org/)》，详情请阅读原著！

## Background
* Gradient Descent旨在朝"下坡"移动，而非明确寻求临界点。而牛顿法的目标是寻求梯度为0的点。
* Gradient Clipping基本思想来源于梯度并没有指明最佳步长，只说明了在无限小区域内的最佳方向。当传统Gradient Descent算法提议更新很大一步时，启发式Gradient Clipping会干涉来减小步长，从而使其不太可能走出梯度近似为最陡下降方向的悬崖区域。
* 假设某个计算图中包含一条反复与矩阵$W$相乘的路径，那么$t$步之后，相当于乘以$W^t$，假设有特征值分解$W=V diag(\lambda)V^{-1}$，在这种情况下，很容易看出：
$$
W^t=(V diag(\lambda)V^{-1})^{t}=V diag(\lambda)^tV^{-1}
$$
因此，当特征值$\lambda_i$ 不在$1$附近时，若在量级上大于1则会出现Gradient Exploding；若小于$1$时，则会出现Gradient Vanishing。

## Basic Algorithm
### SGD
SGD是如今深度学习领域应用非常广泛的一种优化算法，它按照数据生产分布抽取$m$ 个mini-batch (独立同分布)样本，通过计算这些mini-batch的梯度均值，我们可以得到梯度的无偏估计。

![SGD](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-optimization/sgd.jpg)

### Momentum
为了加速训练，Momentum积累了之前梯度指数级衰减的移动平均，并且继续沿该方向移动。Momentum主要目的为了解决Hessian矩阵的病态条件和随机梯度的方差。
$$
v\leftarrow \alpha v-\epsilon \bigtriangledown_{\theta}(\frac{1}{m}\sum_{i=1}^m L(f(x^{(i)};\theta),y^{(i)})\\
\theta\leftarrow \theta + v
$$

![Momentum](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-optimization/momentum.jpg)
SGD中步长只是梯度范数乘以学习率，现在步长取决于梯度序列的大小和排列。当许多连续的梯度指向相同的方向时，步长最大。如果Momentum总是观测到梯度 $g$,那么它会在方向$-g$ 上不停加速，直到达到最终速度，其中步长大小为：
$$
\frac{\epsilon||g||}{1-\alpha}
$$
因此将Momentum超参数视为$\frac{1}{1-\alpha}$有助于理解。例如$\alpha=0.9$ 对应着最大速度10倍于Gradient Descent。

### Nesterov
更新规则如下：
$$
v\leftarrow \alpha v-\epsilon \bigtriangledown_{\theta}(\frac{1}{m}\sum_{i=1}^m L(f(x^{(i)};\theta+\alpha v),y^{(i)})\\
\theta\leftarrow \theta+v
$$
Nesterov和标准Momentum之间的区别在于梯度计算上，Nesterov中，梯度计算在施加Momentum之后。

![Nesterov](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-optimization/nesterov.jpg)

### AdaGrad
Learning rate是一个非常难以调整的超参数之一，如果我们相信方向敏感度在某种程度是轴对齐的，那么每个参数设置不同的学习率，在整个学习过程中自动使用这些学习率是合理的。

AdaGrad是自适应学习率算法的一种。它独立地适应所有模型参数的学习率，缩放每个参数反比于其所有梯度历史平方值总和的平方根。具有损失最大偏导的参数相应地有一个快速下降的学习率，而具有小偏导的参数在学习率上有相对较小的下降。净效果是在参数空间中更为平缓的倾斜方向会取得更大的进步。

![AdaGrad](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-optimization/adagrad.jpg)

### RMSProp
RMSProp修改AdaGrad以在非凸设定下效果更好，改变梯度积累为指数加权的移动平均，AdaGrad旨在应用于凸问题时快速收敛。当应用于非凸函数训练神经网络时，学习轨迹可能穿过了很多不同的结构，最终到达一个局部是凸碗的区域。AdaGrad根据平方梯度的整个历史收缩学习率，可能使得学习率在达到这样的凸结构前就变得太小了。__RMSProp使用指数衰减平均以丢弃遥远过去的历史__，使其能够在找到凸碗结构后快速收敛。

* Standard RMSProp
![RMSProp](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-optimization/rmsprop.jpg)

* RMSProp with Nesterov
![RMSProp with Nesterov](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-optimization/rmsprop_with_nesterov.jpg)

### Adam
在Adam中，动量直接并入了梯度一阶矩(指数加权)的估计。将动量加入RMSProp最直接的方法是将动量应用于缩放后的梯度。结合缩放的动量使用没有明确的理论动机。其次，Adam包括偏置修正，修正从原点初始化的一阶矩(动量项)，和非中心的二阶矩的估计。

![Adam](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-optimization/adam.jpg)

## 二阶近似方法
### 牛顿法
牛顿法是基于二阶泰勒级数展开在某点$\theta_0$附近来近似$J(\theta)$的优化方法，其忽略了高阶导数：
$$
J(\theta)\approx J(\theta_0)+(\theta-\theta_0)^T\bigtriangledown_{\theta} J(\theta_0) + \frac{1}{2}(\theta-\theta_0)^T H(\theta-\theta_0)
$$

更新规则：
$$
\theta^{\star}=\theta_0-H^{-1}\bigtriangledown _{\theta}J(\theta_0)
$$
因此，__对于局部的二次函数(具有正定的$H$)，用$H^{-1}$重新调整梯度，牛顿法会直接跳到极小值__。

![Newton's Method](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-optimization/newtons_method.jpg)

Deep Learning中，Loss Function表明通常是非凸的(有很多特征)，如鞍点。因此使用Newton's Method是有问题的，若Hessian Matrix的特征值并不都是正的，Newton's Method实际上会导致更新朝错误的方向移动。这种情况可以通过正则化Hessian Matrix来避免。常用的正则化策略包括在Hessian Matrix对角线上增加常数$\alpha$。正则化更新变为：
$$
\theta^{\star}=\theta_0-[H(f(\theta_0))+\alpha I]^{-1}\bigtriangledown _{\theta}J(\theta_0)
$$

### 共轭梯度
通过迭代下降的共轭方向以有效避免Hessian Matrix求逆计算的方法。

### BFGS
BFGS是使用矩阵$M_t$近似逆，迭代地低秩更新精度以更好地近似$H^{-1}$。当Hessian逆近似$M_t$更新时，下降方向$\rho_t$为$\rho_t=M_tg_t$。该方向上的线性搜索用于决定该方向上的步长$\epsilon^{\star}$。参数的最后更新为：
$$
\theta_{t+1}=\theta_t + \epsilon^{\star}\rho_t
$$
相比于共轭梯度，BFGS的优点在于其花费较少的时间改进每个线搜索。另一方面，BFGS算法必须存储必须存储Hessian 逆矩阵$M$，需要$O(n^2)$的存储空间，使BFGS不适用于大多数参数巨大的Deep Model。

## 优化策略和元算法
### Batch Normalization
设$H$是需要标准化的某层mini batch激活函数，每个样本的激活出现在矩阵的每一行中。为了标准化$H$，我们将其替换为
$$
H^{'}=\frac{H-\mu}{\sigma}
$$

$$
\mu=\frac{1}{m}\sum_i H_{i,:}
$$

$$
\sigma = \sqrt{\delta+\frac{1}{m}\sum_i (H-\mu)_i^2}
$$
$\delta$是个很小的正值，以避免遇到$\sqrt{z}$的梯度在$z=0$处未定义的问题。

至关重要的是，我们反向传播这些操作，来计算$\mu$和$\sigma$，并应用它们于标准化$H$。这意味着，梯度不会再简单地增加$h_i$的标准差或均值；BatchNorm会消除这一操作的影响，归零其在梯度中的元素。

在测试阶段，$\mu$和$\sigma$可以被替换为训练阶段收集的运行均值。这使得模型可以对单一样本评估，而无需使用定义于整个mini-batch的$\mu$和$\sigma$。