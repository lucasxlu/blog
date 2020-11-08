---
title: "[DL] Knowledge Distillation"
date: 2020-10-18 00:08:34
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



## References
1. Hinton G, Vinyals O, Dean J. [Distilling the knowledge in a neural network](https://arxiv.org/pdf/1503.02531)[J]. arXiv preprint arXiv:1503.02531, 2015.