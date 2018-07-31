---
title: "[ML] Clustering"
date: 2018-07-31 10:22:50
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