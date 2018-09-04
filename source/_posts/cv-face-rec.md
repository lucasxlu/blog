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
![DeepID](https://note.youdao.com/yws/public/resource/d77bdd1365b5f3406a5af65f544c9380/xmlnote/WEBRESOURCEa675c67616b71c50e9d4e77c02129416/7230)

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


## Reference
1. Sun, Yi, Xiaogang Wang, and Xiaoou Tang. ["Deep learning face representation from predicting 10,000 classes."](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.
2. Schroff, Florian, Dmitry Kalenichenko, and James Philbin. ["Facenet: A unified embedding for face recognition and clustering."](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
