---
title: "[ML] Clustering"
date: 2018-07-30 22:01:45
mathjax: true
tags:
- Machine Learning
- Data Science
catagories:
- Algorithm
- Machine Learning
---
## Performance Metric
聚类的性能度量大致有两类。一类是将聚类结果与某个“参考模型”进行比较，称为“外部指标”；另一类是直接考查聚类结果而不利用任何参考模型，称为“内部指标”。

对数据集$D=\{x_1,x_2,\cdots,x_m\}$，假定通过聚类给出的簇划分为$\mathcal{C}=\{C_1,C_2,\cdots,C_k\}$，参考模型给出的簇划分为$\mathcal{C}^{\star}=\{C^{\star}_1,C^{\star}_2,\cdots,C^{\star}_s\}$，相应的，另$\lambda$与$\lambda^{\star}$分别表示$\mathcal{C}$与$\mathcal{C}^{\star}$对应的簇标记向量，我们将样本两两配对考虑，定义：
$$
a=|SS|, SS=\{(x_i,x_j)|\lambda_i=\lambda_j,\lambda_i^{\star}=\lambda_j^{\star},i<j\} \\
b=|SD|, SD=\{(x_i,x_j)|\lambda_i=\lambda_j,\lambda_i^{\star}\neq\lambda_j^{\star},i<j\} \\
c=|DS|, DS=\{(x_i,x_j)|\lambda_i\neq\lambda_j,\lambda_i^{\star}=\lambda_j^{\star},i<j\} \\
d=|DD|, DD=\{(x_i,x_j)|\lambda_i\neq\lambda_j,\lambda_i^{\star}\neq\lambda_j^{\star},i<j\}
$$
其中集合$SS$包含了在$\mathcal{C}$中隶属于相同簇且在$\mathcal{C}^{\star}$中也隶属于相同簇的样本对，集合$SD$包含了在$\mathcal{C}$中隶属于相同簇但在$\mathcal{C}^{\star}$中隶属于不同簇的样本对。易得：
$$
a+b+c+d=\frac{m(m-1)}{2}
$$

基于上式可得以下外部指标：
* Jaccard Coefficient: $JC=\frac{a}{a+b+c}$
* FM Index: $FM=\sqrt{\frac{a}{a+b}\cdot \frac{a}{a+c}}$
* Rand Index: $RI=\frac{2(a+d)}{m(m-1)}$

考虑聚类结果的簇划分$\mathcal{C}=\{C_1,C_2,\cdots,C_k\}$，定义：
* $AVG(C)=\frac{2}{|C|(|C|-1)\sum_{1\leq i<j \leq |C|}} dist(x_i,x_j)$
* $diam(C)=\mathop{max} \limits_{1\leq i < j \leq |C|} dist(x_i,x_j)$
* $d_{min}(C_i,C_j)=\mathop{min} \limits_{x_i\in C_i,x_j\in C_j} dist(x_i,x_j)$
* $d_{cen}(C_i,C_j)=dist(\mu_i,\mu_j)$

基于上式可以推导聚类的内部指标：
* DB Index: $DBI=\frac{1}{k}\sum_{i=1}^k \mathop{max} \limits_{j\neq i} \frac{avg(C_i)+avg(C_j)}{d_{cen}(\mu_i,\mu_j)}$
* Dunn Index: $DI=\mathop{min} \limits_{1\leq i \leq k} \{\mathop{min} \limits_{j\neq i} (\frac{d_{min}(C_i,C_j)}{\mathop{max} \limits_{1\leq l \leq k}diam(C_l)})\}$

## Distance Metric
对于无序属性可采用VDM(Value Difference Metric)，令$m_{u,a}$表示在属性$u$上取值为$a$的样本数，$m_{u,a,i}$表示第$i$个样本在属性$u$上取值为$a$的样本数，$k$为样本数，则属性$u$上两个离散值$a$和$b$之间的VDM为：
$$
VDM_p(a,b)=\sum_{i=1}^k |\frac{m_{u,a,i}}{m_{u,a}}-\frac{m_{u,b,i}}{m_{u,b}}|^p
$$

于是，将$L_P$ Distance和VDM结合可以处理混合属性。假定有$n_c$个有序属性、$n-n_c$个无序属性，则：
$$
MinkovDM_p(x_i,x_j)=\big(\sum_{u=1}^{u_c}|x_{iu}-x_{ju}|^p + \sum_{u=n_c+1}^n VDM_p(x_{iu},x_{ju})\big)^{\frac{1}{p}}
$$
当样本空间中不同属性的重要性不同时，可使用“加权距离”：
$$
dist_{wmk}(x_i,x_j)=\big(w_1\cdot|x_{i1}-x_{j1}|^p + \cdots + w_n\cdot|x_{in}-x_{jn}|^p \big)^{\frac{1}{p}}
$$

## Prototype-based Clustering
### KMeans
KMeans通过最小MSE Loss来进行学习，
$$
E=\sum_{i=1}^k \sum_{x\in C_i}||x-\mu_i||_2^2
$$
其中$\mu_i=\frac{1}{|C_i|}\sum_{x\in C_i}x$是簇$C_i$的均值向量。

### Learning Vector Quantization
LVQ假设数据样本带有类别标记，学习过程中利用样本的这些监督信息来辅助聚类。在学得一组原型向量$\{p_1,p_2,\cdots,p_q\}$后，即可实现对样本$\chi$的簇划分，对任意样本$x$，它将被划入与其距离最近的原型向量所代表的簇中；换言之，每个原型向量$p_i$定义了与之相关的一个区域$R_i$，该区域中每个样本与$p_i$的距离不大于它与其他原型向量$p_{i^{'}}(i^{'}\neq i)$的距离，即：
$$
R_i=\{x\in \chi| ||x-p_i||_2\leq ||x-p_{i^{'}}||_2,i\neq i^{'}\}
$$

### Mixture of Gaussian
与KMeans、LVQ采用原型向量来刻画聚类结构不同，Mixture-of-Gaussian采用概率模型来表达聚类原型。

对$n$维样本空间$\chi$中的随机向量$x$，若$x$服从高斯分布，其概率密度函数为：
$$
p(x)=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}
$$
$\mu$是$n$维均值向量，$\Sigma$是$n\times n$的协方差矩阵。由上式可知，高斯分布完全由均值向量$\mu$和协方差矩阵$\Sigma$确定。我们将概率密度函数记为$p(x|\mu,\Sigma)$。

我们可定义高斯混合分布：
$$
p_{\mathcal{M}}(x)=\sum_{i=1}^k \alpha_i\cdot p(x|\mu_i,\Sigma_i)
$$
该分布共由$k$个混合分布组成，每个混合成分对应一个高斯分布。其中$\mu_i$与$\Sigma_i$是第$i$个高斯混合成分的参数，而$\alpha_i>0$为相应的混合系数，$\sum_{i=1}^k \alpha_i=1$。

假设样本的生成过程由高斯混合分布给出：首先，根据$\alpha_1,\alpha_2,\cdots,\alpha_k$定义的先验分布选择高斯混合成分，其中$\alpha_i$为选择的第$i$个混合成分的概率；然后根据被选择的混合成分的概率密度函数进行采样，从而生成相应的样本。

若training set $D=\{x_1,x_2,\cdots,x_m\}$由上述过程生成，另随机变量$z_j\in \{1,2,\cdots,k\}$表示生成样本$x_j$的高斯混合成分。根据Bayesian Theorem，$z_j$的后验分布对应于：
$$
p_{\mathcal{M}}(z_j=i|x_j)=\frac{P(z_j=i)\cdot p_{\mathcal{M}}(x_j|z_j=i)}{p_{\mathcal{M}}(x_j)}=\frac{\alpha_i \cdot p(x_j|\mu_i,\Sigma_i)}{\sum_{l=1}^k \alpha_l \cdot p(x_j|\mu_l,\Sigma_l)}
$$
$p_{\mathcal{M}}(z_j=i|x_j)$给出了样本$x_j$由第$i$个高斯混合成分生成的后验概率，我们将其记作$\gamma_{ji}$。

当高斯混合分布已知，高斯混合聚类将把样本$D$划分为$k$个簇$\mathcal{C}=\{C_1,C_2,\cdots,C_k\}$，每个样本$x_j$的簇标记如下确定：
$$
\lambda_j=\mathop{argmax} \limits_{i\in \{1,2,\cdots,k\}} \gamma_{ji}
$$

## Density-based Clustering
基于密度的聚类假设聚类结构能通过样本分布的紧密程度确定。DBSCAN是一种著名的密度聚类算法，它基于一组“领域”参数$(\epsilon, MinPts)$来刻画样本分布的紧密程度。给定数据集$D=\{x_1,x_2,\cdots,x_m\}$，定义下面几个概念：
* $\epsilon-$邻域：对$x_j\in D$，其$\epsilon-$邻域包含样本集$D$中与$x_j$的距离不大于$\epsilon$的样本，即$N_{\epsilon}(x_j)=\{x_i\in D|dist(x_i,x_j)\leq \epsilon\}$；
* 核心对象：若$x_j$的$\epsilon-$邻域至少包含$MinPts$个样本，即$|N_{\epsilon}(x_j)|\geq MinPts$，则$x_j$是一个核心对象；
* 密度直达：若$x_j$位于$x_i$的$\epsilon-$邻域内，且$x_i$是核心对象，则称$x_j$由$x_i$密度直达；
* 密度可达：对$x_i$与$x_j$，若存在样本序列$p_1,p_2,\cdots,p_n$，其中$p_1=x_i,p_2=x_j$，且$p_{i+1}$由$p_i$密度直达，则称$x_j$由$x_i$密度可达；
* 密度相连：对$x_i$与$x_j$，若存在$x_k$使得$x_i$与$x_j$均由$x_k$密度可达，则称$x_i$与$x_j$密度相连。

基于以上概念，DBSCAN将“簇”定义为：由密度可达关系导出的最大的密度相连样本集合。形式化的说，给定邻域参数$(\epsilon,MinPts)$，簇$C\subseteq D$是满足以下性质的非空样本子集：
* 连接性：$x_i\in C,x_j \in C, \implies x_i$与$x_j$密度相连
* 最大性：$x_i\in C, x_j$由$x_i$密度可达$\implies$ $x_j \in C$

## Hierarchical Clustering
它先将数据集中的每个样本看作一个初始聚类簇，然后在算法运行的每一步中找出距离最近的两个聚类簇进行合并，该过程不断重复，直到达到预设的聚类簇个数。

## Tips
聚类簇数$k$通常需由用户提供，可运行不同的$k$值后选取最佳的结果。