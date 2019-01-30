---
title: "[DL] Data Augmentation"
date: 2019-01-30 20:48:23
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


## Hide-and-Seek
> Paper: [Hide-and-seek: Forcing a network to be meticulous for weakly-supervised object and action localization](http://openaccess.thecvf.com/content_ICCV_2017/papers/Singh_Hide-And-Seek_Forcing_a_ICCV_2017_paper.pdf)

Hide-and-Seek (HaS)可视为一种提高localisation任务的data augmentation方法，其核心思想也非常简单，即在训练阶段，将input image先划分成$S\times S$个grid，然后随机以概率$p$hidden掉一些grid，来消除DCNN仅仅对image中最discriminative parts的强依赖，而是对relevent parts都产生一定的response，从而提高模型在预测阶段的robustness。

![Main Idea of HaS](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-data-aug/has_main_idea.png)
> 已上图为例，若该image中最discriminative part是dog face，我们在训练阶段从input image中random drop的时候将dog face hidden掉了，那么这样就会迫使模型去寻找其他relevent parts (例如tail和leg)来辅助学习。通过在每个training epoch中随机hidden different parts，模型接受了image不同的parts作为输入，来使得模型关注不同的relevent parts，而不是最discriminative part。

### Delving Into HaS
#### Hiding random image patches
通过randomly hide patches，我们可以保证input image中最discriminative的part并非总是可以被模型get到，既然模型没法总是get到最discriminative的part，那么它自然就会从其他relevent but not that discriminative part中进行学习，从而解决了模型对discriminative part强依赖的缺陷。

具体来讲，是这样的：给定一张size为 $W\times H\times 3$的 training image $I$，我们首先将其划分为固定size的patches ($S\times S$)，然后就得到了一共$(W\times H)/(S\times S)$个patches，然后以概率$p_{hide}$进行patch hiding操作。这样我们就得到了new image $I^{'}$，来作为classification CNN的输入。在test阶段，将整张图作为输入(不做任何hiding)。

![HaS Overview](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-data-aug/has.png)

#### Setting the hidden pixel values
因为training阶段做了patch hiding而test阶段没有，这就存在discrepancy。这就会导致training阶段和test阶段的第一个conv layer activation有不同的distribution。而**一个trained network要拥有良好的泛化能力，activiation的distribution应该是大致一样的**。也就是说，对于DNN中连接到$x$ units的$w$ weights，$w^Tx$的distribution在training/test阶段应该大致相同。然而，在我们的设定中，因一些patch被hidden，而另一些没有被hidden，因此就不能保证activation distribution大致相同了。

假设size为$K\times K$的conv kernel $F$，对应3-dimensional weights $W=\{w_1,\cdots,w_{k\times k}\}$，应用到一个RGB patch $X=\{x_1,\cdots,x_{k\times k}\}$上。另$v$代表每一个hidden pixel的RGB value。我们可以得到3种activation:
1. $F$完全在**visible patch**中，得到输出$\sum_{i=1}^{k\times k} w_i^T x_i$。
2. $F$完全在**hidden patch**中，得到输出$\sum_{i=1}^{k\times k}w_i^T v$。
3. $F$部分**位于visible patch，部分位于hidden patch**，得到输出$\sum_{m\in visible}w_m^Tx_m + \sum_{n\in hidden}w_n^T v$。

在test阶段，$F$总会位于visible patch中(因为HaS只在training阶段work)，因此会输出$\sum_{i=1}^{k\times k}w_i^T x_i$。这种输出仅仅会match到我们上面提到的第一种情况(即$F$完全在**visible patch**中)。对于后面两种情况，training阶段的activation和test阶段的activation也还是不同。

我们通过**将hidden pixel的RGB value设置为整个数据集上图片的mean RGB vector** $v=\mu=\frac{1}{N_{pixels}}\sum_j x_j$来解决以上问题。
其中，$N_{pixels}$代表数据集上的所有像素。

我们不妨来分析一下为什么这能work？
根据期望，一个patch的输出和averaged-valued patch的输出应该是一样的：$\mathbb{E}[\sum_{i=1}^{k\times k}w_i^Tx_i]=\sum_{i=1}^{k\times k}w_i^T\mu$。若将$v$换成$\mu$，那么上面提到的第2种和第3种情况的输出都会是$\sum_{i=1}^{k\times k}w_i^T \mu$，然后就会和test阶段的expected output能match上。

文中用到的检测算法属于weakly-supervised detector (即仅仅给定image的category annotation，不给bbox)，因此整体framework是基于[CAM](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)的，不熟悉的读者可以去阅读[CAM](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)的原文。

### Analysis
* With dropout: 因dropout随机drop掉了RGB channel pixels，image中最discriminative的information依然可以以很高的可能性被模型get到，因此还是会促使模型更多地关注最discriminative的部分。
* GAP VS GMP: 因为GAP促使模型关注**所有的discriminative parts**，而GMP**只关注最discriminative part**。那是否GMP无用呢？实验证明，接入了HaS后的GMP带来了很大的提升，这种improvement可以归因于**max pooling对noise更robust**。



## Reference
1. Zhang, Hongyi, et al. ["mixup: Beyond empirical risk minimization."](https://openreview.net/pdf?id=r1Ddp1-Rb) International Conference on Learning Representations (2018).
2. Kumar Singh, Krishna, and Yong Jae Lee. ["Hide-and-seek: Forcing a network to be meticulous for weakly-supervised object and action localization."](http://openaccess.thecvf.com/content_ICCV_2017/papers/Singh_Hide-And-Seek_Forcing_a_ICCV_2017_paper.pdf) Proceedings of the IEEE International Conference on Computer Vision. 2017.