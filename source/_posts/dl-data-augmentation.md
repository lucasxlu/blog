---
title: "[DL] Data Augmentation"
date: 2019-01-30 20:08:23
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

![Main Idea of HaS](has_main_idea.png)
> 已上图为例，若该image中最discriminative part是dog face，我们在训练阶段从input image中random drop的时候将dog face hidden掉了，那么这样就会迫使模型去寻找其他relevent parts (例如tail和leg)来辅助学习。通过在每个training epoch中随机hidden different parts，模型接受了image不同的parts作为输入，来使得模型关注不同的relevent parts，而不是最discriminative part。

### Delving Into HaS
#### Hiding random image patches
通过randomly hide patches，我们可以保证input image中最discriminative的part并非总是可以被模型get到，既然模型没法总是get到最discriminative的part，那么它自然就会从其他relevent but not that discriminative part中进行学习，从而解决了模型对discriminative part强依赖的缺陷。

具体来讲，是这样的：给定一张size为 $W\times H\times 3$的 training image $I$，我们首先将其划分为固定size的patches ($S\times S$)，然后就得到了一共$(W\times H)/(S\times S)$个patches，然后以概率$p_{hide}$进行patch hiding操作。这样我们就得到了new image $I^{'}$，来作为classification CNN的输入。在test阶段，将整张图作为输入(不做任何hiding)。

![HaS Overview](/has.png)



## Reference
1. Zhang, Hongyi, et al. ["mixup: Beyond empirical risk minimization."](https://openreview.net/pdf?id=r1Ddp1-Rb) International Conference on Learning Representations (2018).
2. Kumar Singh, Krishna, and Yong Jae Lee. ["Hide-and-seek: Forcing a network to be meticulous for weakly-supervised object and action localization."](http://openaccess.thecvf.com/content_ICCV_2017/papers/Singh_Hide-And-Seek_Forcing_a_ICCV_2017_paper.pdf) Proceedings of the IEEE International Conference on Computer Vision. 2017.