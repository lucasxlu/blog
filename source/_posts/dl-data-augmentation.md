---
title: "[DL] Data Augmentation"
date: 2018-11-10 20:08:23
mathjax: true
tags:
- Machine Learning
- Deep Learning
- Data Augmentation
catagories:
- Machine Learning
- Deep Learning
- Data Augmentation
---
## Introduction
众所周知，Deep Learning现如今的繁荣，和**大数据**、**GPU**和**深度学习算法**都是离不开关系的。Deep Learning模型参数众多，需要海量的数据进行拟合，否则很容易overfitting到training set上。而现实情况下，我们不一定能很容易地获取大量高质量标注样本，因此，Data Augmentation则起到了非常大的作用了。这便是本文所要讲述的主角。

## Mixup
> Paper: [mixup: Beyond Empirical Risk Minimization](https://openreview.net/pdf?id=r1Ddp1-Rb)

Mixup的核心idea如下：
$$
\tilde{x}=\lambda x_i + (1-\lambda) x_j
$$

$$
\tilde{y}=\lambda y_i + (1-\lambda) y_j
$$
其中，$x_i, x_j$为raw input vectors，$y_i, y_j$为one-hot encodings。

> Mixup extends the training distribution by incorporating the prior knowledge that linear interpolations of feature vectors should lead to linear interpolations of the associated targets.

Mixup的PyTorch代码如下，是不是非常简洁？
```python
# y1, y2 should be one-hot vectors
for (x1, y1), (x2, y2) in zip(loader1, loader2):
    lam = numpy.random.beta(alpha, alpha)
    x = Variable(lam * x1 + (1. - lam) * x2)
    y = Variable(lam * y1 + (1. - lam) * y2)
    optimizer.zero_grad()
    loss(net(x), y).backward()
    optimizer.step()
```

### What is mixup doing?
The mixup vicinal distribution can be understood as a form of data augmentation that encourages the model $f$ to behave linearly in-between training examples. We argue that this linear behaviour reduces the amount of undesirable oscillations when predicting outside the training examples. Also, linearity is a good inductive bias from the perspective of Occam's razor, since it is one of the simplest possible behaviors.

mixup is a data augmentation method that consists of only two parts: random convex combination of raw inputs, and correspondingly, convex combination of one-hot label encodings.


## Reference
1. Zhang, Hongyi, et al. ["mixup: Beyond empirical risk minimization."](https://openreview.net/pdf?id=r1Ddp1-Rb) International Conference on Learning Representations (2018).