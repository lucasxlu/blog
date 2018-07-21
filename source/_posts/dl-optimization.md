---
title: "[DL] Optimization Algorithm in Deep Learning"
catalog: false
date: 2018-07-20 11:46:37
subtitle:
header-img: "https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-optimization/deep-learning.png"
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
因此将Momentumc超参数视为$\frac{1}{1-\alpha}$有助于理解。例如$\alpha=0.9$ 对应着最大速度10倍于Gradient Descent。

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