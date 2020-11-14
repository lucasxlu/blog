---
title: "[DL] Knowledge Distillation"
date: 2020-11-14 16:15:34
mathjax: true
tags:
- Machine Learning
- Deep Learning
- Model Compression
- Knowledge Distillation
- Data Science
catagories:
- Algorithm
- Machine Learning
- Deep Learning
- Model Compression
- Knowledge Distillation
- Data Science
---
## Introduction
自 AlexNet 以来，Deep Learning 在许多领域取得了突破性的成果。理论上来说，模型的容量（即参数量）越大，性能越好，更能拟合海量数据的分布。与学术界一味追求在某个 benchmark 上刷分突破 SOTA 不同，AI 落地则非常关注能耗与移动端推理。轻量级模型设计与模型压缩也成为了近些年学术界与工业级关注的重心。常见的模型压缩方法有：
* Quantization: 即模型量化，训练时通常采用 ``FP32`` 精度，但部署时可降低精度至 ``FP16`` 甚至 ``Int8``，以此来实现加速的效果
* Network Pruning: 即模型剪枝；因神经网络参数众多，因此模型存在非常大的冗余，可对权重值较小的分支直接删除，然后 ``finetune``；比较有代表性的例子是砍掉 ``VGG`` 网络的 ``fc layers``，并附加 ``GAP layer`` 训练之，参数量能降低一大半，而精度并不会有太大的损失
* Compact Network Design: 即直接设计轻量级模型，代表作是 ``MobileNet`` 和 ``ShuffleNet``，以及近些年比较火热的基于 ``Neural Architecture Search`` 方法直接搜一个最合适的轻量级模型
* Knowledge Distillation: 即本文的主题——知识蒸馏，核心思想是先训练一个性能比较强的大模型，称之为 ``teacher model``，然后来 teach 一个小模型，称之为 ``student model``。通过迫使 ``student model mimic teacher model's behavior``，从而最终让 ``student model`` 接近甚至超过 ``teacher model`` 的精度

关于 ``Quantization/Pruning``，我后面会专门再写两个相关的专题进行介绍，此处不再赘述。关于 ``Compact Network Design``，可参考我之前的文章：[Architecture](https://lucasxlu.github.io/blog/2019/10/20/dl-architecture/)。``Quantization/Pruning`` 通常会带来一定程度的精度损失，即牺牲一定的精度换取推理速度，典型案例是 TikTok 在移动端的部署，人脸生成特效算法 GAN 网络与当前的 SOTA 相比还是相去甚远，但是要让模型在千元机上也能流畅运行起来，从商业角度来说则更为重要。而 ``Knowledge Distillation`` 虽然也有一定程度的精度损失（毕竟小模型的学习能力不如大模型强），但通过算法的改进，有时候小模型甚至能超过大模型的效果。


## What is KD?
提到 KD 就不得不介绍 Hinto 的经典 paper，[Distilling the knowledge in a neural network](https://arxiv.org/pdf/1503.02531)，算得上是 KD 算法的开山之作，核心 idea 也非常简单。以分类问题为例，$z_i$代表 logits，$q_i$ 代表 softmax layer 输出的概率：
$$
q_i=\frac{exp(z_i/T)}{\sum_j exp(z_j/T)}
$$
其中 ``T`` 代表 ``temperature``，若$T=1$，则与原始的 softmax 函数完全一样； ``T`` 值越大 probability distribution 越平滑。最终的 loss function 为 teacher model soft target 与 student model soft target 的 cross entropy loss，以及 student model hard target 与 gt label 的 cross entropy loss 的加权。

> 附加一份 Hinto 的 Slide [Dark Knowledge](https://www.ttic.edu/dl/dark14.pdf)。


## Revisiting Knowledge Distillation via Label Smoothing Regularization
> Paper: [Revisiting Knowledge Distillation via Label Smoothing Regularization](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Revisiting_Knowledge_Distillation_via_Label_Smoothing_Regularization_CVPR_2020_paper.pdf)

这篇 paper 主要探讨了 KD 与 Label Smoothing 的关系，并且通过实验表明了 teacher model 并非一定要比 student model 强，一个性能比较差的 teacher model 也能够带来 performance gain，甚至 student model 来 teach teacher model 也能给 teacher model 带来 performance gain。说明 teacher model 不仅仅能起到提供 similarity information 的作用，还能起到 **regularization** 的作用（即 learnable label smoothing regularization）。基于此，作者提出了一个 teacher-free 的 KD 框架，即 student model 以自己作为 teacher model (aka, self-training)，或者从任意一个 manually-designed regularization distribution 中进行学习，能取得比传统 KD 更佳的学习效果。作者也在 paper 中 challenge 了一把 **a strong teacher model 的必要性**。笔者认为本文还是具备比较高价值的参考意义，下面进行细致讲解。

### LSR
先回顾一下 [Label Smoothing](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf):
一个分类网络在过 softmax layer 之后：
$$
p(k|x)=\frac{exp(z_k)}{\sum_{i=1}^K exp(z_i)}
$$
这里 $z_i$ 为 logit，网络预测 $x$ 为 label $k$ 的概率为 $p(k|x)$，groundtruth label over the distribution 为 $q(k|x)$。对于分类问题，我们通常采用 cross entropy loss 来进行训练: $H(q, p)=-\sum_{k=1}^K q(k|x)log(p(k|x))$，对于 groundtruth label $y$，$q(y|x)=1$；当 $k\neq y$ 时 $q(k|x)=0$。

Label Smoothing 算法修改了 $q(k|x)$ 为 $q^{'}(k|x)$：
$$
q^{'}(k|x)=(1-\alpha)q(k|x) + \alpha u(k)
$$
即 $q^{'}(k|x)$ 成了 $q(k|x)$ 与一个 fixed distribution $\alpha u(k)$ 的加权和。通常地，我们设定 $u(k)$ 为 uniform distribution，即 $u(k)=\frac{1}{K}$。因此，Label Smoothing Cross Entropy 定义为：
$$
H(q^{'},p)=-\sum_{k=1}^K q^{'}(k)log p(k)=(1-\alpha)H(q,p)+\alpha H(u,p)=(1-\alpha)H(q,p)+\alpha (D_{KL}(u,p) + H(u))
$$
$D_{KL}$ 代表 KL-divergence，$H(u)$ 代表 $u$ 的 entropy，是一个固定常数。因此，Label Smoothing 又可以写成：
$$
\mathcal{L}_{LS}=(1-\alpha)H(q,p)+\alpha D_{KL}(u,p)
$$

同样地，在 KD 中，$p_{\tau}^t=softmax(z_k^t)=\frac{exp(z_k^t/\tau)}{\sum_{i=1}^K exp(z_i^t/\tau)}$，$z^t$ 是 teacher model 输出的 logit。KD 的目的就在于 让 student model 通过优化 cross entropy loss 以及 KL divergence 来学习 teacher model 的信息：
$$
\mathcal{L}_{KD}=(1-\alpha)H(q,p)+\alpha D_{KL}(p_{\tau}^t,p_{\tau})
$$

重点来了，看看 $\mathcal{L}_{LS}$ 与 $\mathcal{L}_{KD}$，发现两者的唯一区别在于 $D_{KL}(p_{\tau}^t,p_{\tau})$ 中的 $p_{\tau}^t(k)$ 是来自 teacher model 的 distribution；而 $D_{KL}(u,p)$ 中的 $u(k)$ 是一个 predefined uniform distribution。因此，作者认为 **LSR 是 KD 的一种special case**。并且作者通过实验发现，temperature $\tau$ 越大，$p^t(k)$ 与 label smoothing 中的 uniform distribution $u(k)$ 越相似。

### Self-training
此外，若 teacher model is unavailable 时该如何做 KD 呢？作者还提出了一个 self-training 方案：对于一个模型 $S$，首先按常规方式训练一个pretrained model $S^p$，然后在第二阶段的 KD 过程中用于优化 cross entropy 与 KL divergence:
$$
\mathcal{L}_{self}=(1-\alpha)H(q,p)+\alpha D_{KL}(p_{\tau}^t,p_{\tau})
$$
其中 $p_{\tau}^t,p_{\tau}$ 分别代表模型 $S$ 和 $S^p$ 的 output probability。

第二个 teacher-free KD 方案为手动设定一个100%准确的 virtual teacher model：
$$
p^d(k)=\left\{
\begin{aligned}
a,if && k=c \\
(1-a)/(K-1),if && k\neq c
\end{aligned}
\right.
$$
其中 $K$ 代表类别数量，$c$ 代表正确标签。此时 loss function 定义为：
$$
L_{reg}=(1-\alpha)H(q,p)+\alpha D_{KL}(p_{\tau}^d,p_{\tau})
$$

> 这个算法我自己在项目里也试过，在一些分类任务上确实能带来比较明显的提升，具体实验数据在此不赘述了，请阅读原文。


## References
1. Hinton G, Vinyals O, Dean J. [Distilling the knowledge in a neural network](https://arxiv.org/pdf/1503.02531)[J]. arXiv preprint arXiv:1503.02531, 2015.
2. Yuan L, Tay F E H, Li G, et al. [Revisiting Knowledge Distillation via Label Smoothing Regularization](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Revisiting_Knowledge_Distillation_via_Label_Smoothing_Regularization_CVPR_2020_paper.pdf)[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 3903-3911.