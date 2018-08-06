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
