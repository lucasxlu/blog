---
title: "[CV] Face Recognition"
date: 2018-09-03 18:08:11
mathjax: true
tags:
- Machine Learning
- Deep Learning
- Computer Vision
- Face Recognition
catagories:
- Machine Learning
- Deep Learning
- Computer Vision
- Face Recognition
---
## Introduction
人脸识别(Face Recognition)是工业界和学术界都非常火热的一个方向，并且已经催生出许多成功的应用落地场景，比如刷脸支付、安检等。而Face Recognition最大的突破也是由Deep Learning Architecture + 一系列精巧的Loss Function带来的。本文旨在对Face Recognition领域里的一些经典Paper进行梳理，详情请参阅Reference部分的Paper原文。


## Face Recognition as N-Categories Classification Problems
在Metric Learning里的一系列优秀的Loss还未被引入Face Recognition之前，Face Verification/Identification一个非常直观的想法就是直接train 一个 n-categories classifier。然后将最后一层的输出作为input image的特征，再选取合适的distance metric来决定这两张脸是否属于同一个人。这种做法的一些经典工作就是[DeepID](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)。这篇Paper发表在CVPR2014上面，属于非常古董的模型了，鉴于近年来已经几乎不这么做了，所以本文仅仅象征性地回顾一下这几篇具有代表性的Paper。我们会把讨论重心放在Metric Learning的一系列Loss上。

> Paper: [Deep learning face representation from predicting 10,000 classes](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)

这篇Paper其实idea非常简单，就是把Face Recognition问题转换为一个$N$-类Classification问题，其中$N$代表dataset中identity的数量。为了增强feature representation能力，作者也将各个facial region的特征做concatenation。[DeepID](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)的Architecture如下：
![DeepID(https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-face-rec/deepid.jpg)

注意到DeepID的输入部分，除了后一个conv layer的feature map之外，还有前一个max-pooling的输出，这样做的好处在于Network能够获取multi-scale的input，也可以视为一种skipping layer(将lower level feature和higher level feature做feature fusion)。那么最后一个hidden layer的输入就是这样子的：
$$
y_j=max(0, \sum_i x_i^1\cdot w_{i,j}^1 + \sum_i x_i^2\cdot w_{i,j}^2 + b_j)
$$

另外，作者在实验中意识到，<font color="red">随着identity 数量的增加，整个网络的feature representation learning和performance都会随之增加</font>。[DeepID](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)在LFW上达到了97.45%的精度。


## FaceNet
Google的[FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)是我个人认为在Face Recognition领域里一篇非常insightful的Paper，通过引入triplets并直接在**Euclidean Space**作为feature vector度量，[FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)在LFW上达到了99.63%的效果。

> Paper: [Facenet: A unified embedding for face recognition and clustering.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)

### What is FaceNet?
[FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)的Idea其实也比较简单。简而言之呢，就是通过DNN学习一种**Euclidean Embedding**，来使得inter-class更加compact，inter-class更加地separable，这就是本文的核心角色——[Triplet Loss](https://papers.nips.cc/paper/2795-distance-metric-learning-for-large-margin-nearest-neighbor-classification.pdf)。

![FaceNet](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-face-rec/facenet.jpg)

### Details of Triplet Loss
Triplet Embedding是通过一个Network将输入$x$映射到$d$-维输出$f(x)\in \mathbb{R}^d$，文章中将其做了一个Normalization，即$||f(x)||_2=1$。Triplet Loss的目的就是为了找到一个person的anchor $x_i^a$，使得它与相同identity的positive images $x_i^p$更加地close，而与不同identity的negative images更加地separate。写成公式就是：
$$
||x_i^a-x_i^p||_2^2 + \alpha < ||x_i^a-x_i^n||_2^2 \quad \forall (x_i^a,x_i^p,x_i^n)\in \mathcal{T}
$$
$\alpha$就代表margin，那么Minimization of Triplet Loss就是：
$$
\sum_{i}^N [||f(x_i^a)-f(x_i^p)||_2^2 - ||f(x_i^a)-f(x_i^n)||_2^2 + \alpha ]_+
$$
Triplet Loss确定了，那么下一步就是如何选择合适的Triplets。

### Triplet Selection
<font color="red">为了保证快速收敛，我们需要violate triplet的constraint，即挑选anchor $x_i^a$，来挑选hard positive $x_i^p$来满足$\mathop{argmax} \limits_{x_i^p}||f(x_i^a)-f(x_i^p)||_2^2$，以及hard negative $x_i^n$来满足$\mathop{argmin} \limits_{x_i^p}||f(x_i^a)-f(x_i^n)||_2^2$</font>。
> [@LucasX](https://www.zhihu.com/people/xulu-0620/activities)注：读者仔细体会一下这里和triplet loss definition的区别，为啥是相反的？这里可视为一种[hard negative mining](http://cs.brown.edu/people/pfelzens/papers/lsvm-pami.pdf)。

在整个training set上计算$argmax$和$argmin$是不太现实的，文中采取了两个做法：
* 训练每$n$步离线来生成triplets，使用most recent network checkpoint和dataset的子集来计算$argmax$和$argmin$。
* 在线生成triplets，这种做法可视为在一个mini-batch选择hard positive/negative exemplars。

Selecting the hardest negatives can in practice lead to bad local minima early on in training,specifically it can result in a collapsed model (i.e. $f(x) = 0$). In order to mitigate this, it helps to select $x^n_i$ such that:
$$
||f(x_i^a)-f(x_i^p)||_2^2 < ||f(x_i^a)-f(x_i^n)||_2^2
$$
<font color="red">We call these negative exemplars semi-hard, as they are further away from the anchor than the positive exemplar, but still hard because the squared distance is close to the anchorpositive distance. Those negatives lie inside the margin $\alpha$.</font>

### Experiments
对于Face Verification Task，判断两张图是否为一个人，我们仅需比较这两个特征向量的squared $L_2$ distance $D(x_i,x_j)$是否超过了某个阈值即可。
* True Accepts代表face pairs $(i,j)被正确分类到同一个identity$:
  $TA(d)=\{(i,j)\in \mathcal{P}_{same},\quad with \quad D(x_i,x_j)\leq d\}$
* False Accepts代表face pairs $(i,j)被错误分类到同一个identity$:
  $FA(d)=\{(i,j)\in \mathcal{P}_{diff},\quad with \quad D(x_i,x_j)\leq d\}$


## Center Loss



## Reference
1. Sun, Yi, Xiaogang Wang, and Xiaoou Tang. ["Deep learning face representation from predicting 10,000 classes."](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.
2. Schroff, Florian, Dmitry Kalenichenko, and James Philbin. ["Facenet: A unified embedding for face recognition and clustering."](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
