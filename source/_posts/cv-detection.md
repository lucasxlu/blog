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
下集预告：SPPNet


## Reference
1. Girshick, Ross, et al. ["Rich feature hierarchies for accurate object detection and semantic segmentation."](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.
2. 