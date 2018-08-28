---
title: "[CV] Object Detection"
date: 2018-08-28 11:20:05
mathjax: true
tags:
- Machine Learning
- Deep Learning
- Computer Vision
- Object Detection
catagories:
- Machine Learning
- Deep Learning
- Computer Vision
- Object Detection
---
## Introduction
Object Detection是Computer Vision领域一个非常火热的研究方向。并且在工业界也有着十分广泛的应用(例如人脸检测、无人驾驶的行人/车辆检测等等)。本质旨在梳理RCNN--SPPNet--Fast RCNN--Faster RCNN--FCN--Mask RCNN，YOLO v1/2/3, SSD等Object Detection这些非常经典的工作。

## RCNN (Region-based CNN)
> Paper: [Rich feature hierarchies for accurate object detection and semantic segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)

### What is RCNN?
这篇Paper可以看做是Deep Learning在Object Detection大获成功的开端，对Machine Learning/Pattern Recognition熟悉的读者应该都知道，**Feature Matters in approximately every task!** 而RCNN性能提升最大的因素之一便是很好地利用了CNN提取的Feature，而不是像先前的detector那样使用手工设计的feature(例如SIFT/LBP/HOG等)。

RCNN可以认为是Regions with CNN features，即(1)先利用[Selective Search算法](https://staff.fnwi.uva.nl/th.gevers/pub/GeversIJCV2013.pdf)生成大约2000个Region Proposal，(2)Pretrained CNN从这些Region Proposal中提取deep feature(from pool5)，(3)然后再利用linear SVM进行one-VS-rest分类。从而将Object Detection问题转化为一个Classification问题，对于Selective Search框选不准的bbox，后面使用<font color="orange">Bounding Box Regression</font>(下面会详细介绍)进行校准。这便是RCNN的主要idea。

![RCNN](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-detection/rcnn.png)

### Details of RCNN
#### Pretraining and Fine-tuning
RCNN先将Base Network在ImageNet上train一个1000 class的classifier，然后将output neuron设置为21(20个Foreground + 1个Background)在target datasets上fine-tune。其中，将由Selective Search生成的Region Proposal与groundtruth Bbox的$IOU\geq 0.5$的定义为positive sample，其他的定义为negative sample。

作者在实验中发现，pool5 features learned from ImageNet足够general，并且在domain-specific datasets上学习non-linear classifier可以获得非常大的性能提升。

#### Bounding Box Regression
输入是$N$个training pairs，$\{(P^i,G^i)\}_{i=1,2,\cdots,N}, P^i=(P_x^i,P_y^i,P_w^i,P_h^i)$代表$P^i$像素点的$(x,y)$坐标点、width和height。$G=(G_x,G_y,G_w,G_h)$代表groundtruth bbox。BBox Regression的目的就是为了学习一种mapping使得proposed box $P$ 映射到 groundtruth box $G$。

将$x,y$的transformation设为$d_x(P),d_y(P)$，属于<font color="red">scale-invariant translation。$w,h$是log-space translation</font>。学习完成后，可将input proposal转换为predicted groundtruth box $\hat{G}$:
$$
\hat{G}_x=P_w d_x(P)+P_x
$$

$$
\hat{G}_y=P_h d_x(P)+P_y
$$

$$
\hat{G}_w=P_w exp(d_w(P))
$$

$$
\hat{G}_h=P_h exp(d_h(P))
$$

每个$d_{\star}(P)$都用一个线性函数来进行建模，使用$pool_5$ feature，权重的学习则使用OLS优化即可：
$$
w_{\star}=\mathop{argmin} \limits_{\hat{w}_{\star}} \sum_i^N (t_{\star}^i-\hat{w}_{\star}^T \phi_5 (P^i))^2 + \lambda||\hat{w}_{\star}||^2
$$

The regression targets $t_{\star}$ for the training pair $(P, G)$ are defined as:
$$
t_x=\frac{G_x-P_x}{P_w}
$$

$$
t_y=\frac{G_y-P_y}{P_h}
$$

$$
t_w=log(\frac{G_w}{P_w})
$$

$$
t_h=log(\frac{G_h}{P_h})
$$

在选取Proposed Bbox的时候，我们只选取离Groundtruth Bbox比较近的($IOU\geq 0.6$)来做Bounding Box Regression。

以上就是Deep Learning在Object Detection领域一个开创性的工作--RCNN。若有疑问，欢迎给我留言！


## SPPNet
> Paper: [Spatial pyramid pooling in deep convolutional networks for visual recognition.](https://arxiv.org/pdf/1406.4729v4.pdf)

### What is SPPNet?
SPPNet(Spatial Pyramid Pooling)是基于RCNN进行改进的一个Object Detection算法。介绍SPPNet之前，我们不妨先来看一下RCNN有什么问题？RCNN，即Region-based CNN，它需要CNN作为base network去做特征提取，而传统CNN需要固定的squared input，而为了满足这个条件，就需要手工地对原图进行裁剪、变形等操作，而这样势必会丢失信息。作者意识到这种现象的原因不在于卷积层，而在于FC Layers需要固定的输入尺寸，因此通过在feature map的SSPlayer可以满足对多尺度的feature map裁剪，从而concatenate得到固定尺寸的特征输入。取得了很好的效果，在detection任务上，region proposal直接在feature map上生成，而不是在原图上生成，因此可以仅仅通过一次特征提取，而不需要像RCNN那样提取2000次(2000个 Region Proposal)，这大大加速了检测效率。

> [@LucasX](https://www.zhihu.com/people/xulu-0620/activities)注：现如今的Deep Architecture比较多采用Fully Convolutional Architecture(全卷积结构)，而不含Fully Connected Layers，在最后做分类或回归任务时，采用Global Average Pooling即可。

![Crop/Warp VS SPP](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-detection/cw_vs_spp.jpg)


### Why SPPNet?
SPPNet究竟有什么过人之处得到了Kaiming He大神的赏识呢？
1. SPP可以在不顾input size的情况下获取fixed size output，这是sliding window做不到的。
2. SPP uses multi-level spatial bins，而sliding window仅仅使用single window size。<font color="red">multi-level pooling对object deformation则十分地robust</font>。
3. SPP can pool features extracted at variable scales thanks to the flexibility of input scales.
4. Training with variable-size images increases scale-invariance and reduces over-fitting.

### Details of SPPNet
#### SPP Layer
SPP Layer can maintain spatial information by pooling in local spatial bins. <font color="red">These spatial bins have sizes proportional to the image size, so the number of bins is fixed regardless of the image size.</font> This is in contrast to the sliding window pooling of the previous deep networks,where the number of sliding windows depends on the input size.

![SPP Layer](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-detection/spp_layer.jpg)

这样，通过不同spatial bin pooling得到的k个 M-dimensional feature concatenation，我们就可以得到fixed length的feature vector了，接下来是不是就可以愉快地用FC Layers/SVM等ML算法train了？

#### SPP for Detection
RCNN需要从2K个Region Proposal中feedforwad Pretrained CNN去提取特征，这显然是非常低效的。SPPNet直接将整张image(possible multi-scale))作为输入，这样就可以只feedforwad一次CNN。然后在<font color="red">feature map层面</font>获取candidate window，SPP Layer pool到fixed-length feature representation of the window。

![Pooling](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-detection/pooling.jpg)

Region Proposal生成阶段和RCNN比较相似，依然是[Selective Search](https://staff.fnwi.uva.nl/th.gevers/pub/GeversIJCV2013.pdf)生成2000个bbox candidate，然后将原始image resize使得$min(w, h)=s$，文中采用 4-level spatial pyramid ($1\times 1, 2\times 2, 3\times 3,6\times 6$, totally 50 bins) to pool the features。对于每个window，该Pooling操作得到一个12800-Dimensional (256×50) 的向量。这个向量作为FC Layers的输入，然后和RCNN一样训练linear SVM去做分类。

训练SPP Detector时，正负样本的采样是基于groundtruth bbox为基准，$IOU\geq 0.3$为positive sample，反之为negative sample。


## Fast RCNN
下集预告：Fast RCNN ;-)


## Reference
1. Girshick, Ross, et al. ["Rich feature hierarchies for accurate object detection and semantic segmentation."](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.
2. He, Kaiming, et al. ["Spatial pyramid pooling in deep convolutional networks for visual recognition."](https://arxiv.org/pdf/1406.4729v4.pdf) European conference on computer vision. Springer, Cham, 2014.