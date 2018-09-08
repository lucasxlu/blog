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

> [@LucasX](https://www.zhihu.com/people/xulu-0620/activities)注：本文长期更新。


## Face Recognition as N-Categories Classification Problems
在Metric Learning里的一系列优秀的Loss还未被引入Face Recognition之前，Face Verification/Identification一个非常直观的想法就是直接train 一个 n-categories classifier。然后将最后一层的输出作为input image的特征，再选取合适的distance metric来决定这两张脸是否属于同一个人。这种做法的一些经典工作就是[DeepID](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)。这篇Paper发表在CVPR2014上面，属于非常古董的模型了，鉴于近年来已经几乎不这么做了，所以本文仅仅象征性地回顾一下这几篇具有代表性的Paper。我们会把讨论重心放在Metric Learning的一系列Loss上。

> Paper: [Deep learning face representation from predicting 10,000 classes](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)

这篇Paper其实idea非常简单，就是把Face Recognition问题转换为一个$N$-类Classification问题，其中$N$代表dataset中identity的数量。为了增强feature representation能力，作者也将各个facial region的特征做concatenation。[DeepID](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf)的Architecture如下：
![DeepID](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-face-rec/deepid.jpg)

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
> Paper: [A discriminative feature learning approach for deep face recognition](https://ydwen.github.io/papers/WenECCV16.pdf)

Face Recognition领域，除了设计更加优秀的Network Architecture，也有另一个方向的工作是在设计更加优秀的Loss。[Center Loss](https://ydwen.github.io/papers/WenECCV16.pdf)就是其中之一。和FaceNet中使用[Triplet Loss](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf)的目的一样，[Center Loss](https://ydwen.github.io/papers/WenECCV16.pdf)依然是为了使得intra-class more compact and inter-class more separate。本文就来简要介绍一下[Center Loss](https://ydwen.github.io/papers/WenECCV16.pdf)。

[Center Loss](https://ydwen.github.io/papers/WenECCV16.pdf)通过学习每一个类的中心向量，来同时更新这个center，以及最小化deep features和其对应class的centers之间的距离。CNN的Loss为Softmax Loss与[Center Loss](https://ydwen.github.io/papers/WenECCV16.pdf)的加权。Softmax Loss仅仅会让不同的class分开，但[Center Loss](https://ydwen.github.io/papers/WenECCV16.pdf)还会使得相同class的deep features更加靠近类的centers。通过这种joint supervision(Softmax + Center Loss)，不仅仅inter-class的difference被加大了，而且intra-class的variantions也被减小了。因此便可以学得更加discriminative的feature representation。这便是[Center Loss](https://ydwen.github.io/papers/WenECCV16.pdf)的大致idea。

### What is Center Loss?
Softmax Loss是这样的：
$$
\mathcal{L}_S=-\sum_{i=1}^m log\frac{e^{W_{y_i}^Tx_i+b_{y_i}}}{\sum_{j=1}^n e^{W_j^Tx_i+b_j}}
$$

[Center Loss](https://ydwen.github.io/papers/WenECCV16.pdf)则是这样的：
$$
\mathcal{L}_C=\frac{1}{2}\sum_{i=1}^m ||x_i-c_{y_i}||_2^2
$$
为了更新center vector $c_{y_i}$，文中采取的做法是在每一个mini-batch中进行更新(而不是在整个training set中更新)，然后center vector $c_{y_i}$的计算为相关class feature的平均值。此外，为了避免mislabeled samples，我们使用$\alpha$来控制center vector的learning rate。Center Loss的梯度求导可以表示为：
$$
\frac{\partial \mathcal{L}_C}{\partial x_i}=x_i - c_{y_i}
$$

$$
\Delta c_j=\frac{\sum_{i=1}^m \delta(y_i=j)\cdot (c_j-x_i)}{1+\sum_{i=1}^m\delta(y_i=j)}
$$

where $\delta(condition) = 1$ if the condition is satisfied, and $\delta(condition) = 0$ if not. $\alpha$ is restricted in $[0, 1]$. We adopt the joint supervision of softmax loss and center loss to train the CNNs for discriminative feature learning. The formulation is given in Eq. 5.
$$
\mathcal{L}=\mathcal{L}_S+\lambda \mathcal{L}_C=-\sum_{i=1}^m log\frac{e^{W_{y_i}^Tx_i + b_{y_i}}}{\sum_{j=1}^n e^{W_j^Tx_i + b_j}} + \frac{\lambda}{2} \sum_{i=1}^m ||x_i-c_{y_i}||_2^2
$$

整个学习算法如下：
![Learning of Center Loss](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-face-rec/centerloss_update.jpg)

网络结构如下：
![Center Loss Architecture](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-face-rec/centerloss_nn.jpg)

Center Loss的好处在于：
* Joint supervision of Softmax Loss and Center Loss能够大大加强DCNN的feature learning能力。
* 其他Metric Learning的Loss例如[Triplet Loss](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf), Contractive Loss等pairs selection是非常麻烦的一件事情，但是Center Loss则不需要复杂的triplet pairs selection。

网络学习完成，在做Face Verification/Identification时，<font color="red">第一个 FC Layers的feature被当作特征，同时，我们也将水平翻转图片的feature进行concatenation，作为最终的face feature，PCA降维之后，Cosine Distance, Nearest Neighbor and Threshold comparison用来作为判断是否为同一个人的依据</font>。


## NormFace
> Paper: [Normface: L2 hypersphere embedding for face verification](https://arxiv.org/pdf/1704.06369v4.pdf)




## Reference
1. Sun, Yi, Xiaogang Wang, and Xiaoou Tang. ["Deep learning face representation from predicting 10,000 classes."](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.
2. Schroff, Florian, Dmitry Kalenichenko, and James Philbin. ["Facenet: A unified embedding for face recognition and clustering."](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
3. Wen Y, Zhang K, Li Z, Qiao Y. [A discriminative feature learning approach for deep face recognition](https://ydwen.github.io/papers/WenECCV16.pdf). In European Conference on Computer Vision 2016 Oct 8 (pp. 499-515). Springer, Cham.
4. Wang F, Xiang X, Cheng J, Yuille AL. [Normface: L2 hypersphere embedding for face verification](https://arxiv.org/pdf/1704.06369v4.pdf). InProceedings of the 2017 ACM on Multimedia Conference 2017 Oct 23 (pp. 1041-1049). ACM.
