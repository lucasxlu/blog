---
title: "[DL] Auto Encoder"
date: 2018-08-22 19:17:56
mathjax: true
tags:
- Machine Learning
- Deep Learning
- Data Science
- Auto Encoder
catagories:
- Algorithm
- Machine Learning
- Deep Learning
- Auto Encoder
---
## Introduction
Auto Encoder是深度学习里一个用途非常广的无监督学习模型，常常用来降维或者特征学习(例如我在[豆瓣评论情感挖掘这个repository](https://github.com/lucasxlu/XiaoLuAI/tree/master/nlp)里就使用了Deep AutoEncoder来学习300维word2vec里更discriminative的特征表达)。近年来随着GAN的火热，AutoEncoder也常常站在了generative model的前沿。

## Undercomplete AutoEncoder
从AutoEncoder里获得有用特征的一种方法是限制$h$的维度比$x$小，这种编码维度小于输入维度的AutoEncoder称为Undercomplete AutoEncoder。学习undercomplete representation将强制AutoEncoder捕捉training data中最显著的特征。

当Decoder是线性的且Loss是MSE，Undercomplete AutoEncoder会学习出与PCA相同的生成子空间。这种情况下，AutoEncoder在训练来执行复制任务的同时学到了training data的主元子空间。因此，拥有non-linear Encoder和non-linear Decoder的AutoEncoder能够学习出更强大的PCA非线性推广。

若Encoder和Decoder被赋予过大的容量，AutoEncoder会执行复制任务而捕捉不到任何有关数据分布的有用信息。

## Regularized AutoEncoder
若隐藏编码的维度允许与输入相等，或隐藏编码维数大于输入的overcomplete情况下，即使是linear Encoder和linear Decoder也可以学会将输入复制到输出，而学不到任何有关数据分布的有用信息。

Regularized AutoEncoder使用的Loss Function鼓励模型学习其他特征(除了将输入复制到输出)，而不必限制使用浅层的Encoder和Decoder以及小的编码维数来限制模型容量。

Sparse AutoEncoder简单地在训练时结合Encoder层的稀疏惩罚$\Omega(h)$和重构误差：
$$
L(x,g(f(x)))+\Omega(h)
$$
Sparse AutoEncoder一般用来学习特征。

除了向Cost Function增加一个Regularization，我们也可以通过改变重构误差项来获得一个能学到有用信息的AutoEncoder。

Denoising AutoEncoder最小化：
$$
L(x,g(f(\tilde{x})))
$$
其中$\tilde{x}$是被某种噪声损坏的副本，因此DAE必须撤销这些损坏，而不是简单地复制输入。

另一个Regularized AutoEncoder的策略是使用一个类似Sparse AutoEncoder中的惩罚项$\Omega$，
$$
L(x,g(f(x)))+\Omega(h,x)
$$
但$\Omega$的形式不同：
$$
\Omega(h,x)=\lambda \sum_i ||\triangledown_xh_i||^2
$$
这迫使模型学习一个在$x$变化很小时目标也没有太大变化的函数。因为这个惩罚只对training data适用，它迫使AutoEncoder学习可以反映training data distribution information的特征。这样的正则化AutoEncoder称为Contractive AutoEncoder(CAE)。

## Details of Denosing AutoEncoder
DAE是一类接受损坏数据作为输入，并训练来预测原始未被损坏数据作为输出的AutoEncoder。DAE的训练过程如下：我们引入一个损坏过程$C(\tilde{x}|x)$，这个条件分布代表给定数据样本$x$产生损坏样本$\tilde{x}$的概率。自编码器则根据以下过程，从训练数据对$(x,\tilde{x})$中学习重构分布$p_{reconstruct}(x|\tilde{x})$:
1. 从training data中采一个训练样本$x$
2. 从$C(\tilde{x}|x=x)$采一个损坏样本$\tilde{x}$
3. 将$(x,\tilde{x})$作为训练样本来估计AutoEncoder的重构分布$p_{reconstruct}(x|\tilde{x})=p_{decoder}(x|h)$，其中$h$是Encoder$f(\tilde{x})$的输出，$p_{decoder}$根据解码函数$g(h)$定义。


