---
title: "[CV] Semantic Segmentation"
date: 2018-12-05 20:14:05
mathjax: true
tags:
- Machine Learning
- Deep Learning
- Computer Vision
- Semantic Segmentation
catagories:
- Machine Learning
- Deep Learning
- Computer Vision
- Semantic Segmentation
---
## Introduction
Semantic segmentation也是Computer Vision领域一个非常重要的研究方向，和Classification，Detection一起是high-level vision里最重要的方向。我不是主要做Segmentation的，但由于Segmentation的广泛的应用方向(例如自动驾驶的场景感知)和研究热点，本文旨在梳理近些年CV顶会上一些非常有代表性的work。

> [@LucasX](https://www.zhihu.com/people/xulu-0620/activities)注：本文长期更新。

## Fully Convolutional Networks for Semantic Segmentation
> Paper: [Fully Convolutional Networks for Semantic Segmentation](http://openaccess.thecvf.com/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)

本文针对semantic segmentation问题提出了一个Fully Convolutional Network (FCN)，它可以接受任意size的image作为input，然后给出分割后的预测输出。
> 注：若读者不清楚为啥fully convolutional network能处理arbitrary size input，不妨回想一下经典的detection algorithm **SPPNet**，因为CNN必须接受fixed size作为input的原因就是后面接了几层fully connected layers，如果整个网络都是全卷积的，那自然就不需要了。

此外，因DCNN提取的hierarchical feature，low-level包含更多细节信息，high-level包含更多semantic meaning。Global information解决了"what"，即input image的semantic meaning，而local information解决了"where"，即包含更多的spatial localization information。而根据DCNN提取feature的层次性特点，因此作者在文中很自然引入了skip architecture，把不同level的feature结合起来以提升segmentation performance。

作者使用的Deep Model依然来自于Classification Model (AlexNet/VGG/GoogLeNet)，只是将后面的fully connected layers改成了fully convolutional layers + de-conv来适应segmentation任务。采用反卷积层对最后一个卷积层的feature map进行上采样, 使它恢复到输入图像相同的尺寸，从而可以对每个像素都产生了一个预测, 同时保留了原始输入图像中的空间信息, 最后在上采样的特征图上进行逐像素分类。

### Fully convolutional networks
若FCN的loss为最后一层spatial dimension的求和，$l(x;\theta)=\sum_{ij}l^{'}(x_{ij};\theta)$，那么梯度就可以被计算为每个spatial component的求和。因此**将最后一层的receptive fields作为mini-batch的话，在整张图上SGD优化$l$与spatial component上SGD优化$l^{'}是等效的$**。当这些receptive fileds有大量重叠时，layer-by-layer的feedforward computation/BP 比patch-by-patch的计算要高效。

#### Adapting classifiers for dense prediction
![Convert to FCN](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-segmentation/convert_to_fcn.jpg)

#### Shift-and-stitch is filter rarefaction
> Dense predictions can be obtained from coarse outputs
by stitching together output from shifted versions of the input. If the output is downsampled by a factor of $f$, shift the input $x$ pixels to the right and $y$ pixels down, once for every $(x, y)$ s.t. $0\leq x, y \leq f$. Process each of these $f^2$ inputs, and interlace the outputs so that the predictions correspond to the pixels at the centers of their receptive fields.

> Consider a layer (convolution or pooling) with input stride $s$, and a subsequent convolution layer with filter weights $f_{ij}$ (eliding the irrelevant feature dimensions). Setting the lower layer's input stride to 1 upsamples its output by a factor of $s$. However, convolving the original filter with the upsampled output does not produce the same result as shift-and-stitch, because the original filter only sees a reduced portion of its (now upsampled) input. To reproduce the trick, rarefy the filter by enlarging it as:
$$
f_{ij}^{'}=\begin{cases}
    f_{i/s,j/s} & \text{if $s$ divides both $i$ and $j$}\\
    0 & \text{otherwise}
\end{cases}
$$
> (with $i$ and $j$ zero-based). Reproducing the full net output of the trick involves repeating this filter enlargement layer-by-layer until all subsampling is removed. (In practice, this can be done efficiently by processing subsampled versions of the upsampled input.)

#### Upsampling is backwards strided convolution
> Another way to connect coarse outputs to dense pixels is interpolation. For instance, simple bilinear interpolation computes each output $y_{ij}$ from the nearest four inputs by a linear map that depends only on the relative positions of the input and output cells.

> In a sense, upsampling with factor $f$ is convolution with a fractional input stride of $1/f$. So long as $f$ is integral,a natural way to upsample is therefore backwards convolution (sometimes called deconvolution) with an output stride of $f$. Such an operation is trivial to implement, since it simply reverses the forward and backward passes of convolution. Thus upsampling is performed in-network for end-to-end learning by backpropagation from the pixelwise loss.

#### Patchwise training is loss sampling
> In stochastic optimization, gradient computation is driven by the training distribution. Both patchwise training and fully convolutional training can be made to produce any distribution, although their relative computational efficiency depends on overlap and minibatch size. Whole image fully convolutional training is identical to patchwise training where each batch consists of all the receptive fields of the units below the loss for an image (or collection of images). While this is more efficient than uniform sampling of patches, it reduces the number of possible batches. However, random selection of patches within an image may be recovered simply. Restricting the loss to a randomly sampled subset of its spatial terms (or, equivalently applying a DropConnect mask [36] between the output and the loss) excludes patches from the gradient computation.

> **Sampling in patchwise training can correct class imbalance [27, 7, 2] and mitigate the spatial correlation of dense patches [28, 15]**. In fully convolutional training, class balance can also be achieved by weighting the loss, and loss sampling can be used to address spatial correlation.

### Segmentation Architecture
Base network是由AlexNet/VGG/GoogLeNet改动而来，Loss采用per-pixel multinominal logistic loss。整体architecture如下：
![DAG Nets in FCN](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-segmentation/dag_nets_fcn.jpg)

#### Combining what and where
> We address this by adding skips [1] that combine the final prediction layer with lower layers with finer strides. This turns a line topology into a DAG, with edges that skip ahead from lower layers to higher ones (Figure 3). **As they see fewer pixels, the finer scale predictions should need fewer layers, so it makes sense to make them from shallower net outputs. Combining fine layers and coarse layers lets the model make local predictions that respect global structure**.

> We first divide the output stride in half by predicting from a 16 pixel stride layer. We add a $1\times 1$ convolution layer on top of pool-4 to produce additional class predictions. We fuse this output with the predictions computed on top of conv7 (convolutionalized fc7) at stride 32 by adding a $2\times$ upsampling layer and summing both predictions (see Figure 3). We initialize the $2\times$ upsampling to bilinear interpolation, but allow the parameters to be learned as described in Section 3.3. Finally, the stride 16 predictions are upsampled back to the image. We call this net FCN-16s. FCN-16s is learned end-to-end, initialized with the parameters of the last,coarser net, which we now call FCN-32s. The new parameters acting on pool4 are zeroinitialized so that the net starts with unmodified predictions. The learning rate is decreased by a factor of 100.


## Reference
1. Long J, Shelhamer E, Darrell T. [Fully convolutional networks for semantic segmentation](http://openaccess.thecvf.com/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 3431-3440.