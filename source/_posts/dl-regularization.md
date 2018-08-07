---
title: "[DL] Regularization"
date: 2018-08-06 17:13:07
mathjax: true
tags:
- Machine Learning
- Deep Learning
- Optimization
- Regularization
- Data Science
catagories:
- Algorithm
- Machine Learning
- Deep Learning
- Optimization
- Regularization
---
## Introduction
Regularization是Machine Learning中一个非常重要的概念，是对抗Overfitting最有用的利器。DNN由于参数众多，很容易overfitting，若直接选用small model，则会导致model特征学习、分类能力不足。因此现实中往往是 __使用较大的模型 + 正则化__ 来解决相应问题。因此本文简要介绍一下Deep Learning中常用的正则化方法。

$$
\tilde{J}(\theta;X,y)=J(\theta;X,y)+\alpha\Omega(\theta)
$$

## 参数范数惩罚
我们通常只对weight做惩罚，而不对bias做正则惩罚。精确拟合bias所需的数据通常比拟合weight少得多，每个weight会指定两个变量如何相互作用。而每个bias仅控制一个单变量。这意味着我们不对其进行正则化也不会导致太大的方差。

在DNN中，有时会希望对Network的每一层使用单独的惩罚，并分配不同的$\alpha$系数，而寻找合适的多个超参代价很大。因此为了减少搜索空间，我们会在所有层使用相同的weight decay。

### $L_2$ Regularization
为简单起见，假定DNN中没有bias，因此$\theta$就是$w$。模型Loss Function如下：
$$
\tilde{J}(w;X,y)=\frac{\alpha}{2}w^Tw+J(w;X,y)
$$
与之对应的梯度为：
$$
\bigtriangledown_w \tilde{J}(w;X,y)=\alpha w+\bigtriangledown_w J(w;X,y)
$$
使用SGD更新权重：
$$
w\leftarrow w-\epsilon(\alpha w+\bigtriangledown_w J(w;X,y))
$$
换种写法就是：
$$
w\leftarrow (1-\epsilon\alpha)w-\epsilon\bigtriangledown_w J(w;X,y)
$$
可以看到，加入weight decay后会引起学习规则的修改，即在每一步执行通常的SGD之前会 __先收缩权重向量(将权重向量乘以一个常数因子)__。

以Linear Regression为例，其Cost Function是MSE:
$$
(Xw-y)^T(Xw-y)
$$
我们添加$L_2$ Regularization之后，Cost Function变为:
$$
(Xw-y)^T(Xw-y)+\frac{1}{2}\alpha w^Tw
$$
这将正规方程的解由 $w=(X^TX)^{-1}X^Ty$ 变为 $w=(X^TX+\alpha I)^{-1}X^Ty$。其中，矩阵$X^TX$与协方差矩阵$\frac{1}{m}X^TX$成正比。$L_2$ Regularization将这个矩阵替换为上式中的$(X^TX+\alpha I)^{-1}$，这个新矩阵与原来的是一样的，不同的仅仅是在对角线加了$\alpha$。这个矩阵的对角项对应每个输入特征的方差。我们可以看到，$L_2$ Regularization能让学习算法“感知到”具有较高方差的输入$x$，因此与输出目标的协方差较小(相对增加方差)的特征的权重将会收缩。

### $L_1$ Regularization
为简单起见，假定DNN中没有bias，因此$\theta$就是$w$。模型Loss Function如下：
$$
\tilde{J}(w;X,y)=\alpha||w||_1+J(w;X,y)
$$
对应的梯度:
$$
\bigtriangledown_w \tilde{J}(w;X,y)=\alpha sign(w)+\bigtriangledown_w J(w;X,y)
$$
可与看到，__此时正则化对梯度的影响不再是线性地缩放每个$w_i$，而是添加了一项与$sign(w_i)$同号的常数，使用这种形式的梯度之后，我们不一定能得到$J(X,y;w)$二次近似的直接算数解($L_2$正则化时可以)__。

我们可以将$L_1$ Regularization Cost Function的二次近似分解成关于参数的求和：
$$
\hat{J}(w;X,y)=J(w^{\star};X,y)+\sum_i [\frac{1}{2}H_{i,i}(w_i-w_i^{\star})^2+\alpha |w_i|]
$$
如下形式的解析解(对每一维$i$)可以最小化这个近似Cost Function：
$$
w_i=sign(w_i^{\star})max\{|w_i^{\star}|-\frac{\alpha}{H_{i,i}},0\}
$$
此时： 
1) 若$w_i^{\star}\leq \frac{\alpha}{H_{i,i}}$，正则化后目标中的$w_i$最优值是0。这是因为在方向$i$上$J(w;X,y)$对$\hat{J}(w;X,y)$的贡献被抵消，$L_1$ Regularization将$w_i$推至0。
2) 若$w_i^{\star}> \frac{\alpha}{H_{i,i}}$，正则化不会将$w_i$的最优值推至0，而仅仅在那个方向上移动$\frac{\alpha}{H_{i,i}}$的距离。

相比$L_2$ Regularization，$L_1$ Regularization会产生更稀疏的解(最优值中的一些参数为0)。

### 作为约束的范数惩罚
$$
\theta^{\star}=\mathop{argmin} \limits_{\theta} \mathcal{L}(\theta,\alpha^{\star})=\mathop{argmin} \limits_{\theta} J(\theta;X,y)+\alpha^{\star}\Omega(\theta)
$$
如果$\Omega$是$L_2$范数，那么权重就是被约束在一个$L_2$球中；如果$\Omega$是$L_1$范数，那么权重就是被约束在一个$L_1$范数限制的区域中。

### Data Augmentation
在NN的输入层注入噪声也可以看作Data Augmentation的一种方式。然而，NN对噪声不是很robust。改善NN robustness的方法之一是简单地将随机噪声添加到输入再训练。输入噪声注入是一些Unsupervised Learning Algorithm的一部分（例如Denoise Auto Encoder）。向hidden layer施加噪声也是可行的，这可以被看作在多个抽象层上进行的Data Augmentation。

### Robustness of Noise
对某些模型而言，__向输入添加方差极小的噪声等价于对权重施加范数惩罚__。一般情况下，注入噪声远比简单地收缩参数强大，特别是噪声被添加到hidden units时会更加强大。

### Multi-Task Learning
MTL是通过合并几个任务中的样例(__可以视为对参数施加的软约束__)来提高泛化的一种方式。__当模型的一部分被多个额外的任务共享时，这部分将被约束为良好的值，通常会带来更好的泛化能力__。

### Early Stopping
在训练中只返回使validation set error最低的参数设置，就可以获得使validation set更低的模型(并且因此有希望获得更好的test set error)。在每次validation set有所改善后，我们存储模型参数的副本。当训练算法终止时，我们返回这些参数而不是最新的参数。当validation set error在事先指定的循环次数内没有进一步改善时，算法就会终止。这种策略称为Early Stopping。

对于weight decay，必须小心不能使用太多的weight decay，__以防止网络陷入不良局部极小点__。

Early Stopping需要validation set，这意味着某些training samples不能被输入到模型。为了更好地利用这一额外数据，我们可以在完成Early Stopping的首次训练之后，进行额外的训练。在第二轮，即额外的训练步骤中，所有的training data都会被包括在内。
* 一种策略是再次初始化模型，然后使用所有数据再次训练。在第二轮训练过程中，我们使用第一轮Early Stopping确定的 __最佳Epoch__。
* 另一种策略是保持从第一轮训练获得的参数，__然后使用全部数据继续训练__。在这个阶段，已经没有validation set指导我们需要训练多少步停止。我们可以监控validation set的平均loss，并继续训练，直到它低于Early Stopping终止时的目标值。

![Early Stopping](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-regularization/early_stopping.jpg)

#### 为什么Early Stopping具有Regularization效果？
Bishop __认为Early Stopping可以将优化过程的参数空间限制在初始参数值$\theta_0$的小领域内__。事实上，在二次误差的简单Linear Model和Gradient Descend情况下，我们可以展示Early Stopping相当于$L_2$ Regularization。

![Early Stopping As Regularization](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-regularization/es.jpg)

