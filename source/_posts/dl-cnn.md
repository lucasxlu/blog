---
title: "[DL] CNN"
date: 2018-07-27 16:14:38
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

## Convolution
1. VALID：无论怎样都不使用zero padding，并且filter只允许访问那些图像中能够完全包含整个核的位置。
2. SAME：只进行足够的zero padding来保持输出和输入具有相同的大小；然而输入像素中靠近边缘的部分相比于中间部分对于输入像素的影响更小。这可能会导致边界像素存在一定程度的欠表示。
3. FULL：它进行了足够多的zero padding，使得每个像素在每个方向上恰好被访问了$k$次，最终输出图像的宽度为$m+k-1$。这种情况下，输出像素中靠近边界的部分相比于中间部分是更少像素的函数。这将导致学得一个在卷积特征映射的所有位置都表现不错的单核更为困难。


## Pooling
无论采用什么样的Pooling，当输入做少量变动时，Pooling能够 __帮助输入的表示近似不变__。Shift Invariant指得是当我们对输入进行少量平移时，经过Pooling后的大多数输出并不会发生改变。例如MaxPooling中，Pooling只对周围的最大值比较敏感，而不是对精确的位置。

## Reference
1. [Deep Learning--CNN](https://www.deeplearningbook.org/contents/convnets.html)