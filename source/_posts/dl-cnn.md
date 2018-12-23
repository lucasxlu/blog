---
title: "[DL] CNN"
date: 2018-12-23 20:53:38
mathjax: true
tags:
- Machine Learning
- Deep Learning
- CNN
- Data Science
- Computer Vision
catagories:
- Algorithm
- Machine Learning
- Deep Learning
- CNN
- Computer Vision
---
## Introduction
做Vision的同学们对Convolutional Neural Networks (CNN)一定不会陌生，可以毫不夸张的说，现如今绝大多数的视觉问题，都是由CNN驱动的。从LeNet到如今的ResNet、DenseNet、CliqueNet，CNN的结构发生了很大的变化。本文旨在记录CNN的基础构件。

## Basic of CNN
### Convolution
1. VALID：无论怎样都不使用zero padding，并且filter只允许访问那些图像中能够完全包含整个核的位置。
2. SAME：只进行足够的zero padding来保持输出和输入具有相同的大小；然而输入像素中靠近边缘的部分相比于中间部分对于输入像素的影响更小。这可能会导致边界像素存在一定程度的欠表示。
3. FULL：它进行了足够多的zero padding，使得每个像素在每个方向上恰好被访问了$k$次，最终输出图像的宽度为$m+k-1$。这种情况下，输出像素中靠近边界的部分相比于中间部分是更少像素的函数。这将导致学得一个在卷积特征映射的所有位置都表现不错的单核更为困难。

### Pooling
无论采用什么样的Pooling，当输入做少量变动时，Pooling能够 __帮助输入的表示近似不变__。Shift Invariant指得是当我们对输入进行少量平移时，经过Pooling后的大多数输出并不会发生改变。例如MaxPooling中，Pooling只对周围的最大值比较敏感，而不是对精确的位置。
* Average Pooling通过对邻域内特征数值求平均来实现，能够抑制由于邻域大小受限造成估计值方差增大的现象，```特点是对背景的保留效果更好```。
* Max Pooling通过取邻域内特征最大值来实现，能够抑制网络参数误差造成估计均值偏移的现象，特点是```更好地提取纹理信息```。


## Variants of Convolution
除了传统的卷积，其实还有很多种卷积方式(例如DW Conv/Dilation Conv)，在这里就对笔者认为比较insightful的Conv Filter进行一下讲解。
> [@LucasX](https://www.zhihu.com/people/xulu-0620)注：对于CNN主流的网络结构，可参考我的[这一篇文章](https://lucasxlu.github.io/blog/2018/11/18/dl-architecture/)。

### Dilation Convolution
Dilation conv允许我们在不损失resolution的情况下将multi-scale的information做融合，而multi-scale information对于Classification/Detection/Segmentation都是非常重要的。
> Unlike pyramid-shaped architectures carried over from image classification, the presented context module is designed specifically for dense prediction. It is a rectangular prism of convolutional layers, with no pooling or subsampling. The module is based on dilated convolutions, which support exponential expansion of the receptive field without loss of resolution or coverage.

> Define the receptive field of an element $p$ in $F_{i+1}$ as the set of elements in F0 that modify the value of $F_{i+1}(p)$. Let the size of the receptive field of $p$ in $F_{i+1}$ be the number of these elements. It is easy to see that the size of the receptive field of each element in $F_{i+1}$ is $(2^{i+2}-1)\times (2^{i+2}-1)$. The receptive field is a square of exponentially increasing size.

![Dilation Conv](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-cnn/dilation_conv.png)


## Reference
1. Goodfellow, Ian, et al. [Deep learning](https://www.deeplearningbook.org/contents/convnets.html). Vol. 1. Cambridge: MIT press, 2016.
2. Yu, Fisher, and Vladlen Koltun. ["Multi-scale context aggregation by dilated convolutions."](https://arxiv.org/pdf/1511.07122v3.pdf)//ICLR 2016.