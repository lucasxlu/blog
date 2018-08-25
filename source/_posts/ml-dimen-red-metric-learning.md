---
title: "[ML] Dimension Reduction and Metric Learning"
date: 2018-08-20 22:02:55
mathjax: true
tags:
- Machine Learning
- Data Science
catagories:
- Algorithm
- Machine Learning
---
## Low-Dimension Embedding
在高维情形下出现的数据样本稀疏、距离计算困难等问题，是所有ML算法共同面临的严重障碍，被称为curse of dimensionality。

**Why dimension reduction works?** 
在很多时候，人们观测或收集到的数据样本虽是高维的，但与学习任务密切相关的也许仅仅是某个低维分布，即高维空间中的一个低维Embedding。

### Multiple Dimensional Scaling
MDS算法的目标是获得样本在$d^{'}$维空间的表示$Z\in \mathbb{R}^{d^{'}\times m},d^{'}\leq d$，且任意两个样本在$d^{'}$维空间中的欧氏距离等于原始空间中的距离，即$||z_i - z_j||=dist_{ij}$。

令$B=Z^TZ\in \mathbb{R}^{m\times m}$，其中$B$为降维后样本的内积矩阵，$b_{ij}=z_i^Tz_j$，有
$$
dist_{ij}^2=||z_i||^2+||z_j||^2-2z_i^Tz_j=b_{ii}+b_{jj}-2b_ib_j
$$

令降维后的样本$Z$被中心化，即$\sum_{i=1}^mz_i=0$。显然，矩阵$B$的行和列之和均为0，易知：
$$
\sum_{i=1}^m dist_{ij}^2=tr(B)+mb_{jj} \\
\sum_{j=1}^m dist_{ij}^2=tr(B)+mb_{ii} \\
\sum_{i=1}^m\sum_{j=1}^m dist_{ij}^2=2tr(B)
$$

其中$tr(B)=\sum_{i=1}^m||z_i||^2$，令：
$$
dist_{i\cdot}^2=\frac{1}{m}\sum_{j=1}^m dist_{ij}^2 \\
dist_{\cdot j}^2=\frac{1}{m}\sum_{i=1}^m dist_{ij}^2 \\
dist_{\cdot \cdot}^2=\frac{1}{m}\sum_{i=1}^m \sum_{j=1}^m dist_{ij}^2 \\
$$

可得：
$$
b_{ij}=-\frac{1}{2}(dist_{ij}^2-dist_{i\cdot}^2-dist_{\cdot j}^2+dist_{\cdot \cdot}^2)
$$

由此即可通过降维前后保持不变的距离$D$求取内积矩阵$B$。

对矩阵$B$做特征值分解，$B=V\bigwedge V^T$，其中$\bigwedge=diag(\lambda_1,\lambda_2,\cdots,\lambda_d)$为特征值构成的对角矩阵，$\lambda_1\geq \lambda_2\geq \cdots\geq \lambda_d$，$V$为特征向量矩阵。假定其中有$d^{\star}$个非零特征值，它们构成对角矩阵$\bigwedge_{\star}=diag(\lambda_1,\lambda_2,\cdots,\lambda_{d^{\star}})$，另$V_{\star}$表示相应的特征向量矩阵，则$Z$可表达为：
$$
Z=\bigwedge_{\star}^{1/2}V_{\star}^T\in \mathbb{R}^{d^{\star}\times m}
$$

在现实应用中为了有效降维，往往仅需降维后的距离与原始空间中的距离 **尽可能相近**，而不必严格相等。此时可取$d^{'}\ll d$个最大特征值构成对角矩阵$\tilde{\bigwedge}=diag(\lambda_1,\cdots,\lambda_{d^{'}})$，另$\tilde{V}$表示相应的特征向量矩阵，则$Z$可表达为：
$$
Z=\tilde{\bigwedge}^{1/2}\tilde{V}^T\in \mathbb{R}^{d^{'}\times m}
$$

## PCA
假定数据进行了中心化，即$\sum_i x_i=0$；再假定投影变换后得到的新坐标系为$\{w_1,w_2,\cdots,w_d\}$，其中$w_i$是标准正交基向量，$||w_i||_2=1,w_i^Tw_j=0 (i\neq j)$。若丢弃新坐标系中的部分坐标，即将维度降低到$d^{'}< d$，则样本点$x_i$在低维坐标系中的投影是$z_i=(z_{i1},z_{i2},\cdots,z_{id^{'}})$，其中$z_{ij}=w_j^Tx_i$是$x_i$在低维坐标系下第$j$维的坐标。若基于$z_i$来重构$x_i$，则会得到$\hat{x}_i=\sum_{j=1}^{d^{'}z_{ij}w_j}$。

$$
\min \limits_{W} -tr(W^TXX^TW) \quad s.t. W^TW=I
$$

样本点$x_i$在新空间中超平面上的投影是$W^Tx_i$，若所有样本点的投影能尽可能分开，则应使 **投影后样本点的方差最大**。

即：
$$
\max \limits_{W} tr(W^TXX^TW) \quad s.t. W^TW=I
$$

对上式使用拉格朗日乘子法可得：
$$
XX^TW=\lambda W
$$

于是，只需对协方差矩阵$XX^T$进行特征值分解，将求得的特征值排序：$\lambda_1\geq \lambda_2\geq \cdots \lambda_d$，再取前$d^{'}$个特征值对应的特征向量构成$W=(w_1,w_2,\cdots,w_{d^{'}})$。这就是PCA的解。

降维会导致$d-d^{'}$个特征值的特征向量被舍弃了，这样带来的好处是：一方面，舍弃这部分信息之后能使样本采样密度增大，这正是降维的重要动机；另一方面，当数据受到噪声影响时，最小的特征值所对应的特征向量往往与噪声有关，将它们舍弃能在一定程度上起到去噪的作用。

## Kernel Linear Dimension Reduction
线性降维方法假设从高维空间到低维空间的函数映射是线性的，然而，在不少现实任务中，可能需要非线性映射才能找到恰当的低维嵌入。

非线性降维的一种常见方法是基于Kernel Tricks(读者若对SVM熟悉，此处自然不会陌生了，SVM在对线性不可分的情况下也是利用Kernel Tricks将原始空间映射到高维超平面再基于软间隔最大化去分类)对线性降维方法进行“核化”。

以Kernel PCA为例，假定我们将在高维空间中把数据投影到由$W$确定的超平面上，即PCA欲求解：
$$
(\sum_{i=1}^m z_iz_i^T)W=\lambda W
$$

其中$z_i$是样本点$x_i$在高维特征空间中的像，易知：
$$
W=\frac{1}{\lambda}(\sum_{i=1}^m z_iz_i^T)W=\sum_{i=1}^m z_i \frac{z_i^T W}{\lambda}=\sum_{i=1}^m z_i \alpha_i
$$
其中$\alpha_i=\frac{1}{\lambda}z_i^TW$，假定$z_i$是由原始属性空间中的样本点$x_i$通过映射$\phi$产生，即$z_i=\phi(x_i),i=1,2,\cdots,m$。若$\phi$能被显式表达出来，则通过它将样本映射至高维特征空间，再在特征空间中实施PCA即可。有：
$$
(\sum_{i=1}^m \phi(x_i)\phi(x_i)^T)W=\lambda W
$$

$$
W=\sum_{i=1}^m \phi(x_i) \alpha_i
$$

一般情况下，我们不清楚$\phi$的具体形式，于是引入核函数：
$$
\mathcal{k}(x_i,x_j)=\phi(x_i)^T\phi(x_j)
$$

化简可得：
$$
KA=\lambda A
$$
其中$K$为$\mathcal{k}$对应的核矩阵，$(K)_{ij}=\mathcal{k}(x_i,x_j), A=(\alpha_1;\alpha_2,\cdots,\alpha_m)$。显然，上式是特征值分解问题，取$K$最大的$d^{'}$个特征值对应的特征向量即可。

对新样本$x$，其投影后的第$j (j=1,2,\cdots,d^{'})$维坐标为：
$$
z_j=w_j^T\phi(x)=\sum_{i=1}^m \alpha_i^j \phi(x_i)^T \phi(x) = \sum_{i=1}^m \alpha_i^j \mathcal{k}(x_i, x)
$$
其中$\alpha_i$已经过规范化，$\alpha_i^j$是$\alpha_i$的第$j$个分量。

## Manifold Learning
Manifold是在局部与欧式空间同胚的空间，换言之，<font color="red">它在局部具有欧式空间的性质，能用欧氏距离来进行距离计算</font>。这给降维带来了很大的启发：若低维流形嵌入到高维空间中，则数据样本在高维空间的分布虽然看上去非常复杂，但在<font color="red">局部上仍具有欧式空间的性质</font>。因此可以容易地在局部建立降维映射关系，然后再设法将局部映射关系推广到全局。

### Isometric Mapping
Isomap认为低维流形嵌入到高维空间后，直接在高维空间计算直线距离具有误导性，因为高维空间中的直线距离在低维嵌入流形上是不可达的。

我们可利用<font color="red">流行在局部上与欧式空间同胚</font>这个性质，对每一个点基于欧式距离找出其近邻点，然后就能建立一个近邻连接图，图中近邻点之间存在连接，而非近邻点之间不存在连接。于是，计算两点之间测地线距离问题就转换为计算<font color="red">近邻连接图上两点之间的最短路径</font>问题。

对近邻图的构建通常有两种做法，一种是指定近邻点个数，例如欧氏距离最近的$k$个点为近邻点，这样得到的近邻图称为$k$近邻图；另一种是指定距离阈值$\epsilon$，距离小于$\epsilon$的点被认为是近邻点，这样得到的近邻图称为$\epsilon$近邻图。

### Locally Linear Embedding (LLE)
与Isomap试图保持近邻样本之间的距离不同，<font color="red">LLE试图保持邻域内样本之间的线性关系</font>。假定样本点$x_i$的坐标能通过它的近邻样本$x_j,x_k,x_l$的坐标通过线性组合而重构出来，即：
$$
x_i = w_{ij}x_j + w_{ik}x_k + w_{il}x_l
$$
LLE先为每个样本$x_i$找到其近邻下标集合$Q_i$，然后计算出基于$Q_i$中的样本点对$x_i$进行线性重构的系数$w_i$:
$$
\mathop{min} \limits_{w_1,w_2,\cdots,w_m} \sum_{i=1}^m ||x_i - \sum_{j\in Q_i} w_{ij}x_j||^2_2 \quad \sum_{j\in Q_i}w_{ij}=1
$$
其中$x_i$和$x_j$均为已知，令$C_{jk}=(x_i-x_j)^T(x_i-x_k)$，$w_{ij}$有close-form solution：
$$
w_{ij}=\frac{\sum_{k\in Q_i}C_{jk}^{-1}}{\sum_{l,s\in Q_i}C_{ls}^{-1}}
$$
LLE在低维空间中保持$w_i$不变，于是$x_i$对应的低维空间坐标$z_i$可通过下式求解:
$$
\mathop{min} \limits_{z_1,z_2,\cdots,z_m} \sum_{i=1}^m ||z_i-\sum_{j\in Q_i}w_{ij}z_j||_2^2
$$

令$Z=(z_1,z_2,\cdots,z_m)\in \mathbb{R}^{d^{'}\times m}, (W)_{ij}=w_{ij}$，
$$
M=(I-W)^T(I-W)
$$
则有：
$$
\mathop{min} \limits_{Z} tr(ZMZ^T) \quad s.t. \quad ZZ^T=I
$$
可通过特征值分解求解：$M$最小的$d^{'}$个特征值对应的特征向量组成的矩阵即为$Z^T$。

## Metric Learning
Machine Learning中，对高维数据进行降维的目的是希望找到一个合适的低维空间，在此空间中进行学习能比原始空间性能更好。事实上，每个空间对应了在样本属性上定义的一个距离度量，而寻找合适的空间，实际上就是字寻找一个合适的距离度量。那么为何不直接尝试学习出一个合适的Distance Metric呢？这就是Metric Learning的idea。

对两个$d$维样本$x_i$和$x_j$，它们之间的$L_2$ Distance可表示为：
$$
dist_{ed}^2(x_i,x_j)=||x_i - x_j||_2^2=dist_{ij,1}^2+dist_{ij,2}^2+\cdots +dist_{ij,d}^2
$$
其中$dist_{ij,k}^2$表示$x_i$与$x_j$在第$k$维上的距离。若假定不同属性的重要性不同，则可引入权重$w$，得到：
$$
dist_{wed}^2(x_i,x_j)=||x_i - x_j||_2^2=w_1\cdot dist_{ij,1}^2+w_2\cdot dist_{ij,2}^2+\cdots +w_d\cdot dist_{ij,d}^2=(x_i-x_j)^TW(x_i-x_j)
$$
其中$w_i\geq 0, W=diag(w)$是一个对角矩阵，$(W)_{ii}=w_i$。

我们不仅可以将Error Rate这样的监督学习目标作为度量学习的优化目标，还能在度量学习中引入领域知识。例如，若已知某些样本相似、另一些样本不相似，则可定义must-link约束集合$\mathcal{M}$和cannot-link约束集合$\mathcal{C}$。$(x_i,x_j)\in \mathcal{M}$表示$x_i$与$x_j$相似，$(x_i,x_k)\in \mathcal{C}$表示$x_i$与$x_k$不相似。显然，我们希望相似的样本之间距离较小，不相似的样本之间距离较大。于是可通过求解下面的凸优化问题获得适当的度量矩阵：
$$
\mathop{min} \limits_{M} \quad \sum_{(x_i,x_j)\in \mathcal{M}} ||x_i-x_j||_M^2 \quad s.t. \quad \sum_{(x_i,x_k)\in \mathcal{C}}||x_i-x_k||_M^2\geq 1,M\succeq 0
$$
$M\succeq 0$表明M必须是半正定的。上式要求在不相似样本间的距离不小于1的前提下，使相似样本间的距离尽可能小。

<font color="red">若$M$是一个低秩矩阵，则通过对$M$进行特征分解，总能找到一组正交基，其正交基数目为矩阵$M$的秩$rank(M)$，小于原属性数$d$。于是，Metric Learning学得的结果可衍生出一个降维矩阵$P\in \mathbb{R}^{d\times rank(M)}$，能用于降维的目的</font>。