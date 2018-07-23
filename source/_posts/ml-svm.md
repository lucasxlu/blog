---
title: "[ML] SVM"
mathjax: true
date: 2018-07-22 12:42:39
tags:
- Machine Learning
- Data Science
catagories:
- Algorithm
- Machine Learning
---
## 介绍
SVM是一种非常经典的分类算法，也是很多机器学习面试中必问的算法。它的基本模型是定义在特征空间上的间隔最大的线性分类器。SVM的学习策略就是间隔最大化，等价于 __正则化的Hinge Loss最小化问题__。

当训练数据线性可分时，通过硬间隔最大化学习一个线性的分类器；  
当训练数据近似线性可分时，通过软间隔最大化学习一个线性的分类器；  
当训练数据线性不可分时，通过Kernel Tricks和软间隔最大化学习一个非线性的分类器；

当输入空间为欧式空间或离散集合、特征空间为希尔伯特空间时，核函数表示 __将输入从输入空间映射到特征空间得到的特征向量之间的内积，通过使用核函数可以学习非线性支持的SVM，等价于隐式地在高维的特征空间中学习线性SVM__。

一般的，当training set线性可分时，存在无穷个分离超平面可将两类数据正确分开。MLP利用 __误分类最小策略__，求得分离超平面，不过这时的解有无穷多个。线性可分SVM利用 __间隔最大化__求得分离超平面，这时解是唯一的。

## 线性可分SVM与硬间隔最大化
一般来说，一个点距离分离超平面的远近可以表示分类预测的确信程度。在超平面$w\cdot x+b=0$确定的情况下，$|w\cdot x+b|$能够相对地表示点$x$距离超平面的远近。而$w\cdot x+b$的符号与类标记$y$的符号是否一致能够表示分类是否正确。所以可以用量$y(w\cdot x+b)$来表示分类的正确性及确信程度，此为 __"函数间隔"__。

* 函数间隔：对于给定的训练集T和超平面$(w,b)$，定义超平面$(w,b)$关于样本点$(x_i,y_i)$的函数间隔为:  
  $\hat{\gamma}_i=y_i(w\cdot x_i + b)$

  定义超平面$(w,b)$关于训练集T的函数间隔为超平面$(w,b)$关于T中所有样本点$(x_i,y_i)$的函数间隔的最小值，即：  
  $\hat{\gamma}=\mathop{min} \limits_{i=1,\cdots,N}\hat{\gamma}_i$

  函数间隔可以表示分类预测的正确度及确信度，但是选择分离超平面时，只有函数间隔还不够，因为只要成比例地改变$w$和$b$，超平面没有变，但是函数间隔却变为原来的2倍。这一事实启示我们，可以对分离超平面的法向量$w$加某些约束，如归一化$||w||=1$，使得间隔是确定的。这时函数间隔成为 __几何间隔__。

* 几何间隔：对于给定的训练集T和超平面$(w,b)$，定义超平面$(w,b)$关于样本点$(x_i,y_i)$的几何间隔为：  
  $\gamma_i=y_i(\frac{w}{||w||}\cdot x_i+\frac{b}{||w||})$

  定义超平面$(w,b)$关于训练集T的函数间隔为超平面$(w,b)$关于T中所有样本点$(x_i,y_i)$的函数间隔的最小值，即：  
  $\hat{\gamma}=\mathop{min} \limits_{i=1,\cdots,N}\hat{\gamma}_i$

  超平面$(w,b)$关于样本点$(x_i,y_i)$的几何间隔一般是实例点到超平面的带符号的距离。

  函数间隔和几何间隔有如下关系：  
  $\gamma_i=\frac{\hat{\gamma}_i}{||w||}$

  $\gamma=\frac{\hat{\gamma}}{||w||}$

  __如果$||w||=1，那么函数间隔和几何间隔相等$__。如果超平面参数$w$和$b$成比例地改变(超平面未变)，则函数间隔也按此比例改变，但是几何间隔不变。


最大间隔分离超平面  可以表示为下面的约束最优化问题：  
$$\mathop{max} \limits_{w,b} \gamma s.t.\quad y_i(\frac{w}{||w||}\cdot x_i+\frac{b}{||w||})\geq \gamma,\quad i=1,\cdots,N$$

即我们希望最大化超平面$(w,b)$关于training set的几何间隔$\gamma$，约束条件表示的是超平面$(w,b)$关于每个training sample的几个间隔至少是$\gamma$。

考虑几何间隔和函数间隔的关系，该问题等价于：
$$\mathop{max} \limits_{w,b}\frac{\hat{\gamma}}{||w||} \\
s.t.\quad y_i(w\cdot x_i+b)\geq \hat{\gamma}, \quad i=1,2,\cdots,N$$

最大化$\frac{1}{||w||}$和最小化$\frac{1}{2}||w||^2$是等价的，于是就得到下面的线性可分SVM的最优化问题：
$$\mathop{min} \limits_{w,b}\frac{1}{2}||w||^2 \\
s.t.\quad y_i(w\cdot x_i+b)-1\geq 0, \quad i=1,\cdots,N$$

* 最大间隔分离超平面的存在唯一性：若训练数据集T线性可分，则可将训练集中的样本点完全正确分开的最大间隔分离超平面存在且唯一。

在线性可分情况下，training set的样本点中与分离超平面距离最近的样本点的实例成为支持向量。支持向量是使约束条件等号成立的点，即：
$y_i(w\cdot x_i+b)-1=0$

对$y_i=+1$的正例点，支持向量在超平面 $H_1:w\cdot x+b=1$上，对$y_i=-1$的负例点，支持向量在超平面 $H_2:w\cdot x+b=-1$上。在决定分离超平面时只有支持向量起作用，而其他实例点并不起作用。由于支持向量在确定分离超平面中起着决定性的作用，所以将这种分类模型称为"支持向量机"。


### 学习的对偶算法
为了求解线性可分SVM的最优化问题，将它作为原始最优化问题，应用拉格朗日对偶性，通过求解对偶问题得到原始问题的最优解，这就是线性可分SVM的对偶算法。这样做一来对偶问题更容易求解，二来自然引入Kernel Function，可以扩展到非线性分类问题。

引入拉格朗日乘子$\alpha_i \geq 0, i=1,2,\cdots,N$，定义拉格朗日函数：
$L(w,b,\alpha)=\frac{1}{2}||w||^2-\sum_{i=1}^N \alpha_i y_i(w\cdot x_i + b) + \sum_{i=1}^N \alpha_i$，其中，$\alpha=(\alpha_1,\alpha_2,\cdots,\alpha_N)^T$为拉格朗日乘子向量。

根据拉格朗日对偶性，原始问题的对偶问题是极大极小值问题：
$\mathop{max} \limits_{\alpha} \mathop{min} \limits_{w,b} L(w,b,\alpha)$，所以为了得到对偶问题的解，需要先求$L(w,b,\alpha)$对$w,b$的极小，再求对$\alpha$的极大。

1. 求$\mathop{min} \limits_{w,b} L(w,b,\alpha)$：  
   将拉格朗日函数$L(w,b,\alpha)$分别对$w,b$求偏导，并令其等于0。  
   $\bigtriangledown_wL(w,b,\alpha)=w-\sum_{i=1}^N \alpha_i y_i x_i=0$

   $\bigtriangledown_bL(w,b,\alpha)=\sum_{i=1}^N \alpha_i y_i=0$  
  得:  
  $w=\sum_{i=1}^N\alpha_i y_i x_i$  
  $\sum_{i=1}^N\alpha_i y_i=0$  
  可得:  
  $\mathop{min} \limits_{w,b}L(w,b,\alpha)=-\frac{1}{2}\sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i\cdot x_j) + \sum_{i=1}^N \alpha_i$

2. 求解$\mathop{min} \limits_{w,b} L(w,b,\alpha)$对$\alpha$的极大，即是对偶问题  
  $$\mathop{max} \limits_{\alpha}-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j(x_i\cdot x_j) + \sum_{i=1}^N \alpha_i,\quad s.t. \sum_{i=1}^N \alpha_i y_i=0 \quad \alpha_i \geq 0, i=1,2,\cdots,N $$  

  可转换成下面等价的求极小值的对偶问题：
  $$\mathop{min} \limits_{\alpha}-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j(x_i\cdot x_j) - \sum_{i=1}^N \alpha_i,\quad s.t. \sum_{i=1}^N \alpha_i y_i=0 \quad \alpha_i \geq 0, i=1,2,\cdots,N $$  

#### 线性可分SVM的学习算法
1. 构造并求解约束最优化问题:
  $\mathop{min} \limits_{\alpha} \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i \alpha_j y_i y_j (x_i\cdot x_j)-\sum_{i=1}^N \alpha_i, \quad s.t. \sum_{i=1}^N \alpha_i y_i=0, \alpha_i\geq 0$  
  得到最优解$\alpha^{\star}=(\alpha_1^{\star},\alpha_2^{\star},\cdots,\alpha_N^{\star})^T$。

2. 计算 $w^{\star}=\sum_{i=1}^N\alpha_i^{\star} y_ix_i$ ，并选择 $\alpha^{\star}$ 的一个正分量 $\alpha_j^{\star}>0$，计算:  
  $b^{\star}=y_j-\sum_{i=1}^N\alpha_i^{\star} y_i(x_i\cdot x_j)$

3. 求得分离超平面 $w^{\star}\cdot x+b^{\star}=0$，分类决策函数 $f(x)=sign(w^{\star}\cdot x+b^{\star})$。

## 线性SVM与软间隔最大化
线性不可分意味着某些样本点$(x_i,y_i)$不能满足函数间隔大于等于1的约束条件，为了解决这个问题，可以对每个样本点$(x_i, y_i)$引入一个松弛变量$\xi_i \geq0$，使得函数间隔加上松弛变量大于等于1。这样，约束条件变为:  
$y_i(w\cdot x_i+b)\geq 1-\xi_i$

同时，对每个松弛变量，支付一个代价$\xi_i$，目标函数由原来的$\frac{1}{2}||w||^2$变成 $\frac{1}{2}||w||^2+C\sum_{i=1}^N\xi_i$。这里$C>0$称为惩罚参数，一般由问题决定。$C$值大时对误分类的惩罚加大，$C$值小时对误分类的惩罚变小。最小化Loss有两层含义：使$\frac{1}{2}||w||^2$尽量小即间隔尽量大，同时使得误分类点的个数尽量小，$C$是两者的调和系数。

线性不可分的SVM学习问题变成如下凸二次规划问题：  
$\mathop{min} \limits_{w,b,\xi} \frac{1}{2}||w||^2 + C\sum_{i=1}^N \xi_i \quad s.t. \quad y_i(w\cdot x_i + b)\geq 1-\xi_i, i=1,2,\cdots,N \quad \xi_i \geq0$
   
### 学习的对偶算法
原始问题的对偶问题是:
$$\mathop{min} \limits_{\alpha} \frac{1}{2}\sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j(x_i\cdot x_j)-\sum_{i=1}^N \alpha_i$$
$$s.t. \quad \sum_{i=1}^N \alpha_i y_i=0 \qquad 0\leq\alpha_i \leq C$$

#### 线性可分SVM的学习算法
1. 选择惩罚参数$C>0$，构造并求解凸二次规划问题：  
   $\mathop{min} \limits_{\alpha} \frac{1}{2}\sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j(x_i\cdot x_j)-\sum_{i=1}^N\alpha_i$  
   $s.t. \sum_{i=1}^N\alpha_i y_i=0, \quad 0\leq \alpha_i \leq C$  
   求得最优解$\alpha^{\star}=(\alpha_1^{\star},\alpha_2^{\star},\cdots,\alpha_N^{\star})^T$

2. 计算$w^{\star}=\sum_{i=1}^N\alpha_1^{\star} y_ix_i$  
   选择$\alpha^{\star}$的一个分量$\alpha_j^{\star}$适合条件$0<\alpha_j^{\star}<C$ (通常取所有符合条件的样本点上的均值)，计算  
   $b^{\star}=y_j-\sum_{i=1}^Ny_i \alpha_i^{\star}(x_i\cdot x_j)$
  
3. 求得分离超平面 $w^{\star}\cdot x+b^{\star}=0$，
   分类决策函数$f(x)=sign(w^{\star}\cdot x+b^{\star})$

#### 支持向量
软间隔的支持向量$x_i$或者在间隔边界上，或者在降额边界与分离超平面之间，或者在分离超平面误分一侧。  

若$\alpha_i^{\star}<C$，则$\xi_i=0$，支持向量$x_i$恰好落在间隔边界上；  
若$\alpha_i^{\star}=C, 0<\xi_i<1$，则分类正确，$x_i$将在间隔边界与超平面之间；  
若$\alpha_i^{\star}=C, \xi_i=1$，则$x_i$在分离超平面上；  
若$\alpha_i^{\star}=C, \xi_i>1$，则$x_i$位于分离超平面误分一侧。

#### Hinge Loss
SVM还有另一种解释，即最小化以下Loss Function：  
$\sum_{i=1}^N [1-y_i(w\cdot x_i+b)]_{+} + \lambda ||w||^2$

$L(y(w\cdot x+b))=[1-y(w\cdot x+b)]_{+}$称为 Hinge Loss。这就是说，当样本点$(x_i,y_i)$被正确分类且函数间隔 $y_i(w\cdot x_i+b)$大于1时，损失为0，否则损失是 $1-y_i(w\cdot x_i+b)$。

线性SVM原始最优化问题:  
$\mathop{min} \limits_{w,b,\xi} \frac{1}{2}||w||^2+C\sum_{i=1}^N \xi_i$

$s.t.\quad y_i(w\cdot x_i+b)\geq 1-\xi_i$

$\xi_i\geq 0$

等价于最优化问题：

$\mathop{min} \limits_{w,b} \sum_{i=1}^N [1-y_i(w\cdot x_i+b)]_{+} + \lambda||w||^2$

## 非线性SVM与Kernel Function
设原空间为$\chi \subset R^2, x=(x^{(1)},x^{(2)})^T\in \chi$，新空间为$\mathcal{Z} \subset R^2, z=(z^{(1)},z^{(2)})^T\in \mathcal{Z}$，定义从原空间到新空间的变换(映射)：  
$z=\phi(x)=((x^{(1)})^2,(x^{(2)})^2)^T$  
经过变换$z=\phi(x)$，原空间$\chi \subset R^2$变换为新空间$\mathcal{Z}\subset R^2$，原空间中的点变为新空间中的点。从而原空间线性不可分的情形变为新空间里的线性可分问题。

### kernel Function
设$\chi$是输入空间(欧式空间$R^n$或离散集合)，又设$\mathcal{H}$为特征空间(希尔伯特空间)，若存在一个从$\chi$到$\mathcal{H}$的映射:  
$\phi(x):\chi \to \mathcal{H}$
使得对所有$x,z\in \chi$，函数$K(x,z)$都满足：
$K(x,z)=\phi(x)\cdot \phi(z)$  
则称$K(x,z)$为核函数，$\phi(x)$为映射函数，式子中$\phi(x)\cdot \phi(z)$为$\phi(x)$和$\phi(z)$内积。

Kernel Tricks的想法是，在学习与预测中只定义核函数$K(x,z)$，而不显示地定义映射函数$\phi$，直接计算$K(x,z)$比较容易，而通过$\phi(x)$和$\phi(z)$计算$K(x,z)$并不容易。

Kernel-based SVM等价于经过映射函数$\phi$将原来的输入空间变换到一个新的特征空间，将输入空间中的内积$x_i\cdot x_j$变换为特征空间中的内积$\phi(x_i)\cdot \phi(x_j)$，在新的特征空间里从训练样本中学习线性SVM。当映射函数是非线性函数时，学习到的含有核函数的SVM是非线性分类模型。

Kernel Tricks：学习是隐式地在特征空间进行的，不需要显示地定义特征空间和映射函数，这样的技巧称为Kernel Tricks。

### 常用Kernel Function
1. 多项式核函数：  
   $K(x,z)=(x\cdot z+1)^p$  
   对应的SVM是一个$p$次多项式分类器，在此情形下，分类决策函数成为:  
   $f(x)=sign(\sum_{i=1}^{N_s}a_i^{\star}y_i(x_i\cdot x+1)^p+b^{\star})$

2. 高斯核函数：  
   $K(x,z)=exp(-\frac{||x-z||^2}{2\sigma^2})$
   对应的SVM是RBF分类器，在此情形下，分类决策函数成为:  
   $f(x)=sign(\sum_{i=1}^{N_s}a_i^{\star}y_i exp(-\frac{||x-z||^2}{2\sigma^2})+b^{\star})$

### 非线性SVM
从非线性分类数据集，通过Kernel Tricks与软间隔最大化，或凸二次规划，学习到的分类决策函数：  
$f(x)=sign(\sum_{i=1}^N\alpha_i^{\star}y_i K(x,x_i)+b^{\star})$
称为非线性支持向量，$K(x,z)$是正定核函数。

#### 非线性SVM学习算法
1. 选取适当的核函数$K(x,z)$和适当的参数$C$，构造并求解最优化问题：  
   $\mathop{min} \limits_{\alpha} \frac{1}{2}\sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i,x_j)-\sum_{i=1}^N \alpha_i$

   $s.t. \sum_{i=1}^N \alpha_i y_i=0 \qquad 0\leq \alpha_i \leq C$
   求得最优解$\alpha^{\star}=(\alpha_1^{\star},\alpha_2^{\star},\cdots,\alpha_N^{\star})^T$。

2. 选择$\alpha^{\star}$的一个正分量$0<\alpha_i^{\star}<C$，计算 $b^{\star}=y_j-\sum_{i=1}^N\alpha_i^{\star}y_iK(x_i,x_j)$

3. 构造决策函数：  
   $f(x)=sign(\sum_{i=1}^N\alpha_i^{\star}y_iK(x\cdot x_i)+b^{\star})$