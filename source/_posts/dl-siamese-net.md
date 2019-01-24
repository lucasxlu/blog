---
title: "[DL] Siamese Neural Network"
date: 2019-01-24 20:53:38
mathjax: true
tags:
- Machine Learning
- Deep Learning
- Data Science
- Siamese Neural Network
catagories:
- Algorithm
- Machine Learning
- Deep Learning
- Siamese Neural Network
---
## Introduction
Siamese Network也是一个比较有意思的网络结构，并且在许多领域都有了非常成功的应用，本文主要记录这些具体的application中一些代表性的paper。
> [@LucasX](https://www.zhihu.com/people/xulu-0620)注：对网络结构感兴趣的可以阅读我的另外一篇文章[Architecture](https://lucasxlu.github.io/blog/2018/11/18/dl-architecture/)。

## Siamese Network
> Paper: [Signature verification using a "siamese" time delay neural network](http://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf)

这算是Siamese Network最早的一篇文章了，记录了用Siamese Network做signature verification的应用。整体上比较简单，就记录一下key points吧。
* 本文中用到的Siamese Network是两个identical subnetwork，来从两张input image中提取feature，那么verification就是比较extracted feature和该signer之前保存的signature的feature vector之间的distance。
  
![Base Siamese Network](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-siamese-net/base_siam_net.jpg)
* 网络中所有的weights都是learnable，但是两个subnetwork被限制于**weights都是相同的**。
* 在Testing的时候，只用到其中一个subnetwork的输出作为feature vector，来和stored signature feature vector进行比对distance。


## Siamese neural networks for One-shot Image Recognition
> Paper: [Siamese neural networks for one-shot image recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)





## Reference
1. Bromley, Jane, et al. ["Signature verification using a" siamese" time delay neural network."](http://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf) Advances in neural information processing systems. 1994.
2. Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. ["Siamese neural networks for one-shot image recognition."](http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf) ICML Deep Learning Workshop. Vol. 2. 2015.
3. Leal-Taixé, Laura, Cristian Canton-Ferrer, and Konrad Schindler. ["Learning by tracking: Siamese cnn for robust target association."](https://www.cv-foundation.org//openaccess/content_cvpr_2016_workshops/w12/papers/Leal-Taixe_Learning_by_Tracking_CVPR_2016_paper.pdf) Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2016.
4. [Siamese Networks: Algorithm, Applications And PyTorch Implementation](https://becominghuman.ai/siamese-networks-algorithm-applications-and-pytorch-implementation-4ffa3304c18)