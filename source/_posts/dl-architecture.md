---
title: "[DL] Architecture"
date: 2018-10-23 23:07:40
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

> Given a computational budget, ShuffleNet can use wider feature maps. We find this is critical for small networks, as tiny networks usually have an insufficient number of channels to process the information. In addition, in ShuffleNet depthwise convolution only performs on bottleneck feature maps. Even though depthwise convolution usually has very low theoretical complexity, we find it difficult to efficiently implement on lowpower mobile devices, which may result from a worse computation/memory access ratio compared with other dense operations.

#### Advantages of Point-Wise Convolution
Note that group convolution allows more feature map channels for a given complexity constraint, so we hypothesize that the performance gain comes from wider feature maps which help to encode more information. In addition, a smaller network involves thinner feature maps, meaning it benefits more from enlarged feature maps.

#### Channel Shuffle vs. No Shuffle
The purpose of shuffle operation is to enable cross-group information flow for multiple group convolution layers. The evaluations are performed under three different scales of complexity. It is clear that channel shuffle consistently boosts classification scores for different settings. Especially, when group number is relatively large (e.g. g = 8), models with channel shuffle outperform the counterparts by a significant margin, which shows the importance of cross-group information interchange.


## MobileNet V1
> Paper: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861v1.pdf)

CNN驱动了许多视觉任务的飞速发展，然而传统结构例如ResNet、Inception、VGG等FLOP非常大，这使得对于移动端和嵌入式设备的训练与部署变得非常困难。所以近些年来，轻量级网络的设计也成为了一个非常热门的研究方向，[MobileNet](https://arxiv.org/pdf/1704.04861v1.pdf)和[ShuffleNet](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf)就是其中的代表。前面我们已经介绍了[ShuffleNet](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf)，本篇我们就大致回顾一下[MobileNet](https://arxiv.org/pdf/1704.04861v1.pdf)。

### Depth-wise Separable Convolution
MobileNet最主要的结构就是**Depth-wise Separable Convolution**。DW Conv为什么能减少model size呢？我们不妨先来细致分析一下传统的卷积需要多少参数:
假设传统卷积层接受一个$D_F\times D_F\times M$的feature map作为输入，然后输出$D_F\times D_F\times N$的feature map，所以卷积核的size是$D_K\times D_K\times M\times N$，所以需要的计算量为：$D_K\times D_K\times M\times N\times D_F\times D_F$，所以Computational Cost依赖于input channel $M$，output channel $N$，卷积核尺寸$D_K\times D_K$和feature map的尺寸$D_F\times D_F$。

但是MobileNet应用Depth-wise Conv来对Kernel Size和Output Channel进行了解耦。传统Conv Operation通过filters来对features进行filter，然后重组(Combinations)以形成新的representations。Filtering和Combinations可通过DW Separable Conv来分成两步进行。

**Depth-wise Separable Convolution由两层组成：Depth-wise Conv + Point-wise Conv**。DW Conv对于每个channel应用单个filter，PW Conv (a simple $1\times 1$ Conv)用来建立DW layer输出的linear combination。**DW Conv的Computational Cost为: $D_K\times D_K\times M\times D_F\times D_F$**，与传统Conv相比低了$N$倍。

DW Conv虽然高效，然而它仅仅filter了input channel，__却没有对其进行combination__，所以需要额外的layer来对DW Conv后的feature进行linear combination来产生新的representation，这就是基于$1\times 1$ Conv的PW Conv。

The combination of depthwise convolution and $1\times 1$ (pointwise) convolution is called depthwise separable convolution.

DW Separable Conv的Computational Cost为：
$D_K\times D_K \times M\times D_F \times D_F + M\times N\times D_F\times D_F$，前者为DW Conv的cost，后者为PW Conv的cost。

By expressing convolution as a two step process of filtering and combining we get a reduction in computation of:
$$
\frac{D_K\times D_K \times M\times D_F \times D_F + M\times N\times D_F\times D_F}{D_K\times D_K \times M\times N\times D_F \times D_F}=\frac{1}{N} + \frac{1}{D_K^2}
$$
可以看到，Depth-wise Separable Conv的计算量仅仅为传统Conv的$\frac{1}{N} + \frac{1}{D_K^2}$。
![DW Separable Conv](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-architecture/dw-sep-conv.png)

### Width Multiplier: Thinner Models
In order to construct these smaller and less computationally expensive models we introduce a very simple parameter $\alpha$ called width multiplier. The role of the width multiplier α is to thin a network uniformly at each layer. The computational cost of a depthwise separable convolution with width multiplier $\alpha$ is:

$D_K\times D_K\times \alpha M\times D_F\times D_F +\alpha M\times \alpha N + D_F\times D_F$

where $\alpha\in (0, 1]$ with typical settings of 1, 0.75, 0.5 and 0.25. $\alpha = 1$ is the baseline MobileNet and $\alpha < 1$ are reduced MobileNets. Width multiplier has the effect of reducing computational cost and the number of parameters quadratically by roughly $\alpha^2$ . Width multiplier can be applied to any model structure to define a new smaller model with a reasonable accuracy, latency and size trade off. It is used to define a new reduced structure that needs to be trained from scratch.

### Resolution Multiplier: Reduced Representation
The second hyper-parameter to reduce the computational cost of a neural network is a resolution multiplier $\rho$. We apply this to the input image and the internal representation of every layer is subsequently reduced by the same multiplier. In practice we implicitly set $\rho$ by setting the input resolution. We can now express the computational cost for the core layers of our network as depthwise separable convolutions with width multiplier $\alpha$ and resolution multiplier $\rho$:
 
$D_K\times D_K\times \alpha M\times \rho D_F\times \rho D_F +\alpha M\times \alpha N + \rho D_F\times \rho D_F$

where $\rho\in (0, 1]$ which is typically set implicitly so that the input resolution of the network is 224, 192, 160 or 128. $\rho = 1$ is the baseline MobileNet and ρ < 1 are reduced computation MobileNets. Resolution multiplier has the effect of reducing computational cost by $\rho^2$.


## MobileNet V2
> Paper: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf)

[MobileNet V2](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf)是发表在[CVPR2018](http://openaccess.thecvf.com/CVPR2018.py)上的Paper，在移动端网络的设计方面又向前走了一步。MobileNet V2最大的contribution如下: 

> Our main contribution is a novel layer module: the inverted residual with linear bottleneck. This module takes as an input a low-dimensional compressed representation which is first expanded to high dimension and filtered with a lightweight depthwise convolution. Features are subsequently projected back to a low-dimensional representation with a linear convolution.

### Preliminaries, discussion and intuition
#### Depthwise Separable Convolutions
说起Light-weighted Architecture呢，DW Conv是必不可少的组件了，同理，MobileNet V2也是基于DW Conv做的改进。

> The basic idea is to replace a full convolutional operator with a factorized version that splits convolution into two separate layers. The first layer is called a depthwise convolution, it performs lightweight filtering by applying a single convolutional filter per input channel. The second layer is a $1\times 1$ convolution, called a pointwise convolution, which is responsible for building new features through computing linear combinations of the input channels.

传统Conv接受一个$h_i\times w_i\times d_i$输入，应用卷积核$K\in \mathcal{R}^{k\times k\times d_i\times d_j}$来产生$h_i\times w_i\times d_j$的输出。所以传统Conv的Computational Cost为：$h_i\times w_i\times d_i \times d_j\times k\times k$。而DW Separable Conv的Computational Cost仅仅为：$h_i\times w_i\times d_i(k^2+d_j)$，__减少了将近$k^2$倍的计算量__。

#### Linear Bottlenecks
> It has been long assumed that manifolds of interest in neural networks could be embedded in low-dimensional subspaces. In other words, when we look at all individual d-channel pixels of a deep convolutional layer, the information encoded in those values actually lie in some manifold, which in turn is embeddable into a low-dimensional subspace.

DCNN的架构大致是这样的：Conv + ReLU + (Pool) + (FC) + Softmax。DNN之所以拟合能力超强，原因就在于non-linearity transformation，而由于gradient vanishing/exploding的原因，Sigmoid已经淡出了历史舞台，取而代之的是ReLU。我们来分析分析ReLU有什么缺点：

> It is easy to see that in general if a result of a layer transformation ReLU(Bx) has a non-zero volume S, the points mapped to interior S are obtained via a linear transformation B of the input, thus indicating that the part of the input space corresponding to the full dimensional output, is limited to a linear transformation. **In other words, deep networks only have the power of a linear classifier on the non-zero volume part of the output domain.**

> On the other hand, when ReLU collapses the channel, it inevitably loses information in that channel. However if we have lots of channels, and there is a a structure in the activation manifold that information might still be preserved in the other channels. In supplemental materials, we show that if the input manifold can be embedded into a significantly lower-dimensional subspace of the activation space then the ReLU transformation preserves the information while introducing the needed complexity into the set of expressible functions.

简而言之，ReLu有以下两种性质：
1. 若Manifold of Interest在ReLU之后非零，那么它就相当于是一个线性变换。
2. ReLU能够保存input manifold完整的信息，**但是当且仅当input manifold位于input space的低维子空间中时**。

所以，也就不难理解为什么MobileNet V2要先将high-dimensional hidden representations先做一次low-dimensional embedding，然后再变换回到high-dimensional了。

![Evolution of Separable Conv](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-architecture/evolution-of-separable-conv.png)

#### Inverted residuals
> The bottleneck blocks appear similar to residual block where each block contains an input followed by several bottlenecks then followed by expansion [8]. However, inspired by the intuition that the bottlenecks actually contain all the necessary information, while an expansion layer acts merely as an implementation detail that accompanies a non-linear transformation of the tensor, we use shortcuts directly between the bottlenecks.

回想一下，[MobileNet V1](https://arxiv.org/pdf/1704.04861v1.pdf)的结构就是一个普通的feedforwad network，而shortcut在[ResNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)里已经被证明是非常effective的了，所以MobileNet V2自然而然地引入了skip connection了。

#### MobileNet V2 Architecture
> We use ReLU6 as the non-linearity because of its robustness when used with low-precision computation [27]. We always use kernel size $3\times 3$ as is standard for modern networks, and utilize dropout and batch normalization during training.

![DCNN Architecture Comparison](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-architecture/mobilenetv2-cnn-comparison.png)


## ResNeXt
> Paper: [Aggregated Residual Transformations for Deep Neural Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf)

### Introduction
自AlexNet以来，Deep Learning涌现了一大批设计优良的网络(如VGG，Inception，ResNet等)。ResNeXt则在ResNet的基础上，一定程度上参考了Inception的设计，即**split-transform-merge**。


### The Core of ResNeXt
> Unlike VGG-nets, the family of Inception models [38, 17, 39, 37] have demonstrated that carefully designed topologies are able to achieve compelling accuracy with low theoretical complexity. The Inception models have evolved over time [38, 39], but an important common property is a split-transform-merge strategy. In an Inception module, the input is split into a few lower-dimensional embeddings (by 1×1 convolutions), transformed by a set of specialized filters (3×3, 5×5, etc.), and merged by concatenation. It can be shown that the solution space of this architecture is a strict subspace of the solution space of a single large layer (e.g., 5×5) operating on a high-dimensional embedding. The split-transform-merge behavior of Inception modules is expected to approach the representational power of large and dense layers, but at a considerably lower computational complexity.

尽管Inception的**split-transform-merge**策略是非常行之有效的，但是该网络结构过于复杂，人工设计的痕迹过重(相比之下VGG和ResNet则是由相同的block stacking而成)，给人的感觉就是专门为了ImageNet去做的优化，所以当你想要迁移到其他的dataset时就会比较麻烦。因此，ResNeXt的设计就是：在VGG/ResNet的stacking block的基础上，融合进了Inception的split-transform-merge策略。这就是ResNeXt的基础idea。作者在实验中发现cardinality (the size of the set of transformations)对performance的影响是最大的，甚至要大于width和depth。

> Our method harnesses additions to aggregate a set of transformations. But we argue that it is imprecise to view
our method as ensembling, because the members to be aggregated
are trained jointly, not independently.

> The above operation can be recast as a combination of
splitting, transforming, and aggregating. 
1. Splitting: the vector $x$ is sliced as a low-dimensional embedding, and in the above, it is a single-dimension subspace $x_i$. 
2. Transforming: the low-dimensional representation is transformed, and in the above, it is simply scaled: $w_i x_i$.
3. Aggregating: the transformations in all embeddings are aggregated by $\sum_{i=1}^D$.

若将$W$更换为更一般的形式，即任意一种function mapping: $\mathcal{T}(x)$，那么aggregated transformations就变成了:
$$
\mathcal{F}(x)=\sum_{i=1}^C \mathcal{T}_i(x)
$$
其中$\mathcal{T}_i$可以将$x$映射到低维空间。$C$是transformation的size，也就是本文主角——**cardinality**。

> In Eqn.(2), $C$ is the size of the set of transformations
to be aggregated. We refer to $C$ as cardinality [2]. In
Eqn.(2) $C$ is in a position similar to $D$ in Eqn.(1), but $C$
need not equal $D$ and can be an arbitrary number. While
the dimension of width is related to the number of simple
transformations (inner product), we argue that the dimension
of cardinality controls the number of more complex
transformations. We show by experiments that cardinality
is an essential dimension and can be more effective than the
dimensions of width and depth.

![ResNeXt Block](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-architecture/resnext_block.jpg)

那么在ResNet的identical mapping背景下，就变成了这样一种熟悉的结构：
$$
y=x+\sum_{i=1}^C \mathcal{T}_i(x)
$$

> **Relation to Grouped Convolutions**. The above module becomes more succinct using the notation of grouped convolutions [24]. This reformulation is illustrated in Fig. 3(c). All the low-dimensional embeddings (the first $1\times 1$ layers) can be replaced by a single, wider layer (e.g., $1\times 1$, 128-d in Fig 3(c)). Splitting is essentially done by the grouped convolutional layer when it divides its input channels into groups. The grouped convolutional layer in Fig. 3(c) performs 32 groups of convolutions whose input and output channels are 4-dimensional. The grouped convolutional layer concatenates them as the outputs of the layer. The block in Fig. 3(c) looks like the original bottleneck residual block in Fig. 1(left), except that Fig. 3(c) is a wider but sparsely connected module.

![Equivalent Building Blocks of ResNeXt](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-architecture/equivalent_building_blocks_of_resnext.jpg)


## DenseNet
> Paper: [Densely Connected Convolutional Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)

### What is DenseNet?
DenseNet是CVPR2017 Best Paper，是继ResNet之后更加优秀的网络。[前面我们已经介绍过](https://lucasxlu.github.io/blog/2018/10/23/dl-architecture/#ResNet)，ResNet一定程度上解决了gradient vanishing的问题，通过ResNet中的identical mapping使得网络深度可以到达上千层。那么DenseNet又做了哪些改进呢？本文为你一一解答！

在介绍DenseNet之前，我们先回顾一下ResNet做了什么改动，当shortcuts还未被引入DCNN之前，AlexNet/VGG/GoogLeNet都属于构造比较简单的feedforward network，即信息**一层一层往前传播，在BP时梯度一层一层往后传**，但是这样在网络结构很深的时候，就会存在gradient vanishing的问题。所以Kaiming He创造性地引入了skip connection，来使得信息可以从第$i$层之间做identical mapping传播到第$i+t$层，这样就保证了信息的高效流通。
> 注: 关于ResNet更详细的介绍，请参考[这里](https://lucasxlu.github.io/blog/2018/10/23/dl-architecture/#ResNet)。

![Dense Block](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-architecture/dl-architecture/dense_block.jpg)

而DenseNet，就是把这种skip connection做到了极致。为了保证信息在不同layer之间的流通，DenseNet将skip connection做到了每一层和该层之后的所有层中。和ResNet中采用的DW Summation不同的是，DenseNet直接concatenate不同层的features (因为ResNet中DW Summation会影响信息流动)。

值得一提的是，**DenseNet比ResNet的参数更少**。因为dense connection的结构，使得网络不需要重新学习多余的feature maps。
此外，every layer都可以从loss层直接获得梯度，从early layer是获得信息流，也产生了一种**deep supervision**。最后，作者还注意到dense connections一定程度上可以视为Regularization，所以可以缓解overfitting。

它有以下优点：
* 减轻了gradient vanishing问题
* 加强了梯度传播
* 更好的feature reuse
* 极大地减少了参数

### Delve into DenseNet
#### Dense connectivity
To further improve the information flow between layers we propose a different connectivity pattern: we introduce direct connections from any layer to all subsequent layers. Figure 1 illustrates the layout of the resulting DenseNet schematically. Consequently, the ℓth layer receives the feature-maps of all preceding layers, $x_0, \cdots, x_{l-1}$, as input:
$$
x_l=H_l([x_0,x_1,\cdots,x_{l-1}])
$$
where $[x_0,x_1,\cdots,x_{l-1}]$ refers to the concatenation of the feature-maps produced in layers $0, \cdots, l−1$.

#### Pooling layers
The concatenation operation used in Eq. (2) is not viable when the size of feature-maps changes. However, an essential part of convolutional networks is down-sampling layers that change the size of feature-maps. To facilitate down-sampling in our architecture we divide the network into multiple densely connected dense blocks; see Figure 2. We refer to layers between blocks as transition layers, which do convolution and pooling. The transition layers used in our experiments consist of a batch normalization layer and an $1\times 1$ convolutional layer followed by a $2\times 2$ average pooling layer.

#### Growth rate
If each function $H_l$ produces $k$ featuremaps, it follows that the $l$-th layer has $k_0 + k\times (l−1)$ input
feature-maps, where $k_0$ is the number of channels in the input layer. An important difference between DenseNet and
existing network architectures is that DenseNet can have
very narrow layers, e.g., $k = 12$. We refer to the hyperparameter $k$ as the growth rate of the network. We show in Section 4 that a relatively small growth rate is sufficient to obtain state-of-the-art results on the datasets that we tested on.

**One explanation for this is that each layer has access
to all the preceding feature-maps in its block and, therefore,
to the network's "collective knowledge". One can view the
feature-maps as the global state of the network. Each layer
adds $k$ feature-maps of its own to this state. The growth
rate regulates how much new information each layer contributes
to the global state. The global state, once written,
can be accessed from everywhere within the network and,
unlike in traditional network architectures, there is no need
to replicate it from layer to layer**.

#### Bottleneck layers
Although each layer only produces $k$ output feature-maps, it typically has many more inputs. It has been noted in [36, 11] that a $1\times 1$ convolution can be introduced as bottleneck layer before each $3\times 3$ convolution to reduce the number of input feature-maps, and thus to improve computational efficiency. We find this design especially effective for DenseNet and we refer to our network with such a bottleneck layer, i.e., to the BN-ReLU-Conv($1\times 1$)-BN-ReLU-Conv($3\times 3$) version of $H_l$, as DenseNet-B. In our experiments, we let each $1\times 1$ convolution produce 4k feature-maps.


## Reference
1. Krizhevsky A, Sutskever I, Hinton G E. [Imagenet classification with deep convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)[C]//Advances in neural information processing systems. 2012: 1097-1105.
2. Simonyan K, Zisserman A. [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556v6.pdf)[J]. arXiv preprint arXiv:1409.1556, 2014.
3. Lin M, Chen Q, Yan S. [Network in network](https://arxiv.org/pdf/1312.4400v3.pdf)[J]. arXiv preprint arXiv:1312.4400, 2013.
4. He K, Zhang X, Ren S, Sun J. [Deep residual learning for image recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf). InProceedings of the IEEE conference on computer vision and pattern recognition 2016 (pp. 770-778).
5. Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxiao and Sun, Jian. [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf)[C]//The IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018
6. Chollet, Francois. ["Xception: Deep Learning with Depthwise Separable Convolutions."](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf) 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2017.
7. Howard, Andrew G., et al. ["Mobilenets: Efficient convolutional neural networks for mobile vision applications."](https://arxiv.org/pdf/1704.04861v1.pdf) arXiv preprint arXiv:1704.04861 (2017).
8. Zhu, Mark Sandler Andrew Howard Menglong, and Andrey Zhmoginov Liang-Chieh Chen. ["MobileNetV2: Inverted Residuals and Linear Bottlenecks."](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf)[C]//The IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2018
9. Xie, Saining, et al. ["Aggregated residual transformations for deep neural networks."](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf) Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on. IEEE, 2017.
10. Huang, Gao, et al. ["Densely Connected Convolutional Networks."](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) CVPR. Vol. 1. No. 2. 2017.

