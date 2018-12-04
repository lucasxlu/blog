---
title: "[Book] ML Practice At MeiTuan"
date: 2018-12-04 20:15:11
mathjax: true
tags:
- Book Notes
- Machine Learning
- Deep Learning
catagories:
- Book Notes
- Machine Learning
- Deep Learning
---
## Introduction
本文主要记录一下最近在看的一本书[《美团机器学习实践》](https://book.douban.com/subject/30243136/)，这本书和我们所熟知的PRML/MLAPP/DL Book等不同的是，它非常偏实践，而非理论，所以可作为参考书使用。一千个读者会产生一千个哈姆雷特，对这本书的评价也看读者自己吧，个人给3.5/5 $\star$，模型层面上可能没有太大的新意，但是结合了美团的具体业务场景，所以还是推荐阅读。


## 问题建模
### 评估指标
#### 分类指标
$$
Precision=\frac{TP}{TP+FP}
$$

$$
Recall=\frac{TP}{TP+FN}
$$

$$
Accuracy=\frac{TP+TN}{TP+FP+FN+TN}
$$

$$
Error Rate=\frac{FP+FN}{TP+FP+FN+TN}
$$

ROC曲线的纵轴是TPR，横轴是FPR。
$$
TPR=\frac{TP}{TP+FN}
$$

$$
FPR=\frac{FP}{FP+TN}
$$

#### 回归指标
* MAE
* Weighted MAE
  $$
  WMAE=\frac{1}{N}\cdot \sum_{i=1}^N w_i|y_i-p_i|
  $$
* Mean Absolute Percentage Error (MAPE)
  $$
  MAPE=\frac{100}{N}\cdot \sum_{i=1}^N |\frac{y_i-p_i}{y_i}|, y_i\neq 0
  $$
  > MAPE通过计算绝对误差百分比来表示预测效果，其取值越小越好。若MAPE=10，说明预测值平均偏离真实值10%。
* RMSE
* Root Mean Squared Logarithmic Error (RMSLE)
  $$
  RMSLE=\sqrt{\frac{1}{N}\sum_{i=1}^N (log(y_i+1)-log(p_i+1))^2}
  $$
  > RMSLE对预测值偏小的样本惩罚比对预测值偏大的样本惩罚更大。若评价指标采用RMSLE，没法直接优化RMSLE，所以通常会先对预测目标进行对数变换$y_{new}=log(y+1)$，最后预测值再还原为 $p=exp(p_{new})-1$

#### 排序指标
* Mean Average Precision (MAP)
  $$
  AP@K=\frac{\sum_{k=1}^{min(M,K)}P(k)\cdot rel(k)}{min(M,K)}
  $$
  $$
  P(i)=\frac{前i个结果中相关文档数量}{i}
  $$
  > 其中，$AP@K$ 代表计算前 $K$ 个结果的平均准确率；$M$ 代表每次排序的文档总数，可能一次返回文档数不足 $K$ 个；$P(k)$ 表示前$K$ 个结果是否是相关文档，相关取1，不相关取0。

  $$
  MAP@K=\sum_{q=1}^Q \frac{AP_q@K}{Q}
  $$
  其中$Q$ 为查询数量，$AP_q@K$ 为第$q$ 次查询的 $AP@K$ 结果。

* Normalized Discounted Cumulative Gain (NDCG)
  $$
  DCG@K=\sum_{k=1}^K \frac{2^{rel_k}-1}{log_2(k+1)}
  $$

  $$
  IDCG@K=\sum_{k=1}^{|REL|}\frac{2^{rel_k}-1}{log_2(K+1)}
  $$

  $$
  NDCG@K=\frac{DCG@K}{IDCG_K}
  $$
  其中，$NDCG@K$ 表示计算前$K$ 个结果的NDCG，$rel_k$ 表示第$k$ 个位置的相关性得分，$IDCG@K$ 是前$K$ 个排序返回结果能得到的最佳排序结果，用于归一化$DCG@K$，$|REL|$为结果集按相关性排序后的相关性得分列表。

## Model Ensemble
### 理论分析
#### 模型多样性度量
假设binary classification分类结果如下：
|  | $h_i=1$ | $h_i=0$ |
| :---: | :---: | :---: |
| $h_j=1$ | a | c |
| $h_j=0$ | b | d |

* 不一致度量
  $$
  dis_{i,j}=\frac{b+c}{m}
  $$

* 相关系数
  $$
  \rho_{ij}=\frac{ad-bc}{\sqrt{(a+b)(a+c)(c+d)(b+d)}}
  $$

* Q统计
  $$
  Q_{ij}=\frac{ad-bc}{ad+bc}
  $$

* K统计
  $$
  K=\frac{p_1-p_2}{1-p_2}
  $$

  $$
  p_1=\frac{a+d}{m}
  $$

  $$
  p_2=\frac{(a+b)(a+c) + (c+d)(b+d)}{m^2}
  $$

* 双次失败度量
  $$
  df_{ij}=\frac{\sum_{k=1}^m \prod (h_i(x_k)\neq y_k \bigwedge h_j(x_k)\neq y_k)}{m}
  $$

* KW 差异
  $$
  KW=\frac{1}{mT^2}\sum_{k=1}^m l(x_k)(T-l(x_k))
  $$

* k 度量
  $$
  k=1-\frac{1}{(T-1)\bar{p}(1-\bar{p})}KW
  $$

  $$
  \bar{p}=\frac{1}{mT}\sum_{i=1}^T \sum_{k=1}^m \prod (h_i(x_k)=y_k)
  $$

* 熵度量
  $$
  Ent_{cc}=\frac{1}{m}\sum_{k=1}^m \sum_{y\in \{-1, 1\}} -P(y|x_k)log P(y|x_k)
  $$

  $$
  P(y|x_k)=\frac{1}{T}\sum_{i=1}^T \prod (h_i(x_k)=y_k)
  $$


## Feature Engineering
当获取到一批数据后，可以先做**探索性数据分析**，即可视化/统计分析基础统计量，来探索数据内部的规律。

对那些目标变量为输入特征的光滑函数模型，例如Linear Regression/Logistic Regression，其对输入特征的大小很敏感，因此需要做归一化。而那些Tree-based model，如Random Forest/GBDT等，其对输入特征的大小不敏感，因此不需要归一化。

对于**数值**特征，通常有如下处理方法：
* 截断
* 二值化
* bin
* scale
* 缺失值处理
    * 对于缺失值，填充mean
    * 对于异常值，填充median
    * XGBoost可以自动处理缺失feature
* 特征交叉：即让特征直接进行四则运算来获取新的特征(例如$area=width\times height$)，**特征交叉可以在linear model中引入非线性性质，从而提升模型的表达能力**
* 非线性编码：t-SNE，局部线性embedding，谱embedding
* 行统计量：mean/median/mode/maximum/minimum/偏度/峰度

对于**类别**特征，通常有如下处理方法：
* 自然数编码
* one-hot encoding
* 分层编码
* 散列编码
* 计数编码：将类别特征用其对应的计数来代替
* 计数排名编码

### Filter
* Coverage：计算每个特征的coverage(特征在training set中出现的比例)，若feature的coverage很小，则此coverage对模型的预测作用不大，可以剔除。
* Pearson Correlation：计算两个变量$X$和$Y$直接的相关性：
    $$
    \rho_{X,Y}=\frac{cov(X,Y)}{\alpha_X \alpha_Y}=\frac{E[(X-\mu_X)(Y-\mu_Y)]}{\alpha_X \alpha_Y}
    $$

    $$
    r=\frac{\sum_{i=1}^n (X_i-\bar{X})(Y_i-\bar{Y})}{\sqrt{\sum_{i=1}^n (X_i-\bar{X})^2} \sqrt{\sum_{i=1}^n (Y_i-\bar{Y})^2}}
    $$
* Fisher score：对于分类问题，好的feature应该是在同一个category中的取值比较类似，而在不同category之间的取值差异比较大。
    $$
    S_i=\frac{\sum_{j=1}^K n_j(\mu_{ij}-\mu_i)^2}{\sum_{j=1}^K n_j \rho_{ij}^2}
    $$
    其中，$\mu_{ij}$和$\rho_{ij}$分别为特征$i$在类别$j$中的mean和variance。Fisher score越高，特征在不同类别直接的差异性越大，在同一类别中的差异性越小，则该特征越重要。
* 假设检验：$\chi^2$统计量越大，特征相关性越高。
* Mutual Information：MI越大表明两个变量相关性越高。
    $$
    MI(X,Y)=\sum_{y\in Y}\sum_{x\in X}p(x,y)log(\frac{p(x,y)}{p(x)p(y)})=D_{KL}(p(x,y)||p(x)p(y))
    $$
* 最小冗余最大相关性(mRMR)：由于单变量filter方法只考虑了单特征变量和目标变量之间的相关性，因此选择的特征子集可能过于冗余。mRMR在进行feature selection时考虑了feature之间的冗余性，对跟已选择feature的相关性较高的冗余feature进行惩罚。
* 相关特征选择(CFS)：好的feature set包含跟目标变量非常相关的feature，但这些feature之间彼此不相关，对于包含$k$个feature的集合，CFS定义如下：
    $$
    CFS=\mathop{max} \limits_{S_k}[\frac{r_{cf_1} + r_{cf_2} + \cdots + r_{cf_k}}{\sqrt{k + 2(r_{f_1f_2} + \cdots + r_{f_if_j} + \cdots + r_{f_kf_l})}}]
    $$

### Bagging
Bagging直接使用ML算法评估特征子集的效果，它可以检测出两个或多个feature之间的交互关系，而且选择的特征子集让模型的效果达到最优。Bagging是特征子集搜索 + 评估指标 相结合的方法，前者提供候选的新特征子集，后者则基于新特征子集训练一个模型，并用validation set进行评估，为每一组特征子集进行打分，然后选择最优的特征子集。
* 完全搜索
* 启发式搜索(greed search)：feedforwad search/backward search/feedforwad + backward search
* 随机搜索

### Embedding
即与具体的ML算法结合。
* LASSO：L1 Regularization可一定程度上做feature selection
* Decision Tree
* SVM
* Random Forest
* GBDT


