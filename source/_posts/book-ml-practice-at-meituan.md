---
title: "[Book] ML Practice At MeiTuan"
date: 2018-11-24 19:05:11
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