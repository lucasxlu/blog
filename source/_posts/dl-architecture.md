---
title: "[DL] Architecture"
date: 2018-09-07 15:29:40
mathjax: true
tags:
- Machine Learning
- Deep Learning
- Computer Vision
- Image Classification
- Network Architecture
catagories:
- Machine Learning
- Deep Learning
- Computer Vision
- Image Classification
- Network Architecture
---
## Introduction
Deep Learning有三宝：Network Architecture，Loss Function and Optimization。对于大多数人而言，Optimization是搞不动的，所以绝大多数的Paper偏向还是设计更好的Network Architecture或者堆更加精巧的Loss Function。Ian Goodfellow大佬也曾说过：现如今Deep Learning的繁荣，网络结构探究的贡献度远远高于优化算法的贡献度。所以本文旨在梳理从AlexNet到CliqueNet这些经典的work。
> [@LucasX](https://www.zhihu.com/people/xulu-0620)注：对于优化算法，可参考我的[这一篇文章](https://lucasxlu.github.io/blog/2018/07/20/dl-optimization/)。

## AlexNet
> Paper: [Imagenet classification with deep convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

AlexNet可以看作是Deep Learning在Large Scale Image Classification Task上第一次大放异彩。也是从AlexNet起，越来越多的Computer Vision Researcher开始将重心由设计更好的hand-crafted features转为设计更加精巧的网络结构。因此AlexNet是具备划时代意义的经典work。

AlexNet整体结构其实非常非常简单，5层conv + 3层FC + Softmax。AlexNet使用了ReLU来代替Sigmoid作为non-linearity transformation，并且使用双GPU训练，以及一系列的Data Augmentation操作，Dropout，对于今天的工作仍然具备很深远的影响。

## VGG
> Paper: [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556v6.pdf)

VGG也是一篇非常经典的工作，并且在今天的很多任务上依旧可以看到VGG的影子。不同于AlexNet，VGG使用了非常小的Filter($3\times 3$)，以及类似于<font color="red">basic block</font>的结构(读者不妨回想一下GoogLeNet、ResNet、ResNeXt、DenseNet是不是也是由一系列block堆积而成的)。

不妨思考一下为啥要用$3\times 3$的卷积核呢？
1. 两个堆叠的$3\times 3$卷积核对应$5\times 5$的receptive field。而三个$3\times 3$卷积核对应$7\times 7$的receptive field。那为啥不直接用$7\times 7$卷积呢？原因就在于通过堆叠的3个$3\times 3$卷积核，<font color="red">我们引入了更多的non-linearity transformation，这有助于我们的网络学习更加discriminative的特征表达</font>。
2. 减少了参数：3个channel为$C$的$3\times 3$卷积的参数为: $3(3^2C^2)=27C^2$。而一个channel为$C$的$7\times 7$卷积的参数为: $7^2C^2=49C^2$。
    > This can be seen as imposing a regularisation on the $7\times 7$ conv. filters, forcing them to have a decomposition through the $3\times 3$ filters (with non-linearity injected in between)


## ResNet
> Paper: [Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

作者认为，ResNet可以称得上是自AlexNet以来，Deep Learning发展最insightful的idea，ResNet的主角shortcut至今也被广泛应用与Deep Architecture的设计中(如DenseNet, CliqueNet, Deep Layer Aggregation等)。
此前的网络设计趋势是“越来越深”，但神经网络的设计真的就如同段子所言“一层一层往上堆叠就好了吗？”显然不是的，ResNet作者Kaiming He大神在Paper中做了一些实验，验证了当Network越来越深时，Accuracy就饱和了，然后迅速下降，值得一提的是<font color="red">这种性能下降并不是由于参数过多随之而来的overfitting造成的</font>。

### What is Residual Network?
设想DNN的目的是为了学习某种function $\mathcal{H}(x)$，作者并没有直接设计DNN Architecture去学习这种function，而是先学习另一种function $\mathcal{F}(x):=\mathcal{H}(x) - x$，那么原来的$\mathcal{H}(x)$是不是就可以表示成<font color="red">$\mathcal{F}(x)+x$</font>。作者假设这种结构比原先的$\mathcal{H}(x)$更容易优化。
> 例如，若某种identity mapping是最优的，那么，将残差push到0要比通过一系列non-linearity transformation来学习identity mapping更为高效。
 
Shortcut可以表示成如下结构：
$$
y=\mathcal{F}(x, \{W_i\}) + x
$$
$\mathcal{F}(x, \{W_i\})$可以表示多个conv layers，两个feature map通过channel by channel element-wise 叠加。

网络结构的设计方面，依旧是参考了著名的[VGG](https://arxiv.org/pdf/1409.1556v6.pdf)，即：使用大量$3\times 3$ filters并且遵循这两条原则：
1. 对于输出相同feature map size的层使用相同数量的filter
2. 若feature map size减半，则filter的数量则翻倍，来维持每一层的time complexity

对于feature map dimension相同的情况，则只需要element-wise addition即可；若feature map dimension double了，可以采取zero padding来增加dimension，或者采用$1\times 1$ conv来进行升维。

ResNet到这里基本就介绍完了，实验部分当然是在classification/detection/segmentation task上吊打了当前所有的state-of-the-art。ResNet很简单的idea对不对？不得不佩服一下Kaiming大神，他的东西总是简单而有效！


## ShuffleNet
在Computer Vision领域，除了像AlexNet、VGG、GoogLeNet、ResNet、DenseNet、CliqueNet等一系列比较“重量级”的网络结构之外，也有一些非常轻量级的模型，而轻量级模型对于移动设备而言无疑是非常重要的。这里就介绍一下轻量级模型的代表作之一：[ShuffleNet](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf)。
> Paper: [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf)

### What is ShuffleNet?
[ShuffleNet](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf)结构最重要的两个部分就是**Pointwise Group Convolution**和**Channel Shuffle**。

![Channel Shuffle](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-architecture/channel_shuffle.jpg)

![ShuffleNet Unit](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-architecture/shufflenet_unit.jpg)



## Reference
1. Krizhevsky A, Sutskever I, Hinton G E. [Imagenet classification with deep convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)[C]//Advances in neural information processing systems. 2012: 1097-1105.
2. Simonyan K, Zisserman A. [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556v6.pdf)[J]. arXiv preprint arXiv:1409.1556, 2014.
3. Lin M, Chen Q, Yan S. [Network in network](https://arxiv.org/pdf/1312.4400v3.pdf)[J]. arXiv preprint arXiv:1312.4400, 2013.
4. He K, Zhang X, Ren S, Sun J. [Deep residual learning for image recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf). InProceedings of the IEEE conference on computer vision and pattern recognition 2016 (pp. 770-778).
5. Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxiao and Sun, Jian. [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf)[C]//The IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018

