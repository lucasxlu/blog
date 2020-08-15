---
title: "[DL] GAN"
date: 2020-08-15 12:59:50
mathjax: true
tags:
- Machine Learning
- Deep Learning
- GAN
- Data Science
catagories:
- Algorithm
- Machine Learning
- Deep Learning
- GAN
---
## Introduction
GAN (Generative Adversarial Nets)是当前DL/CV方向的研究热门，在工业界也开始有一些比较成功的落地应用，例如TikTok上人脸卡通脸生成、变老特效等等。本文就来介绍一下GAN方向一些非常有代表性的paper。


## GAN
在GAN框架下，会同时训练一个generative model $G$（用来capture data distribution）和discriminative model $D$（用来判断sample来自于真实数据而非$G$生成的），**$G$网络的优化目标在于最大化$D$犯错的概率**，有点类似于两者互相博弈，该模型存在唯一解，即$G$恰好完美拟合了真实数据分布，而$D$恰好无法判断数据到底来自于$G$生成，还是真实数据（分类器输出的softmax score=1/2）。在原始的GAN paper里，$G$和$D$都是MLP，后来经过一系列的改进，结构也变得越来越复杂（例如``DCGAN,StarGAN,BigGAN``）。

为了学习generator distribution在数据集$z$上的分布$p_g$，定义input noise variables $p_z(z)$，input noise space到data space的mapping为$G(z;\theta_g)$；接下来定义discriminator $D$，$D(x)$代表$x$来自于真实数据而非$p_g$的概率。$D$的优化目标在于正确分类数据到底是来自于真实数据还是生成的数据、且最大化分类正确的概率。

同时训练$G$来最小化$log(1 − D(G(z)))$，GAN的优化目标为：
$$
\min_{G}\max_{D}V(D, G)=\mathbb{E}_{x\sim p_{data}(x)}[logD(x)] + \mathbb{E}_{z\sim p_{z}(z)}[log(1-D(G(z)))]
$$
在实际训练中，会先训练$k$个epoch来优化$D$，再训练1个epoch来优化$G$，这样$D$会接近最优解，而$G$参数更新更平缓、稳定。


## DCGAN
前面提到过，原始的GAN paper里，generator和discriminator都是MLP，而DCGAN则引入了CNN结构，作者发现，DCGAN中的generator和discriminator都学习到了hierarchy信息，且可以用作feature extractor。此外，作者对模型进行了如下改进：
- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures. 
- Use ReLU activation in generator for all layers except for the output, which uses Tanh. 
- Use LeakyReLU activation in the discriminator for all layers.



## References
1. Goodfellow I, Pouget-Abadie J, Mirza M, et al. [Generative adversarial nets](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)[C]//Advances in neural information processing systems. 2014: 2672-2680.
2. Radford A, Metz L, Chintala S. [Unsupervised representation learning with deep convolutional generative adversarial networks](https://arxiv.org/pdf/1511.06434.pdf%C3)[J]. arXiv preprint arXiv:1511.06434, 2015.
3. Zhu J Y, Park T, Isola P, et al. [Unpaired image-to-image translation using cycle-consistent adversarial networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2223-2232.