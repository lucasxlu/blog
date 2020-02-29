---
title: "[CV] Classification"
date: 2020-02-29 15:17:05
mathjax: true
tags:
- Machine Learning
- Deep Learning
- Computer Vision
- Visual Classification
catagories:
- Machine Learning
- Deep Learning
- Computer Vision
- Visual Classification
---
## Introduction
Visual Classification是CV领域最最重要的fundamental task，没有之一，并且是其他task (例如RCNN-based detection)的基础。关于分类，现如今主流方法是设计更加精巧的网络结构（请参阅[DL-Architecture](https://lucasxlu.github.io/blog/2019/10/20/dl-architecture/)），或直接NAS搜一个，或设计更有效的Loss Function辅助model learning（请参阅[ML-Loss Function](https://lucasxlu.github.io/blog/2018/07/24/ml-loss/)）。尽管已经发展很成熟，但在实际应用场景中，依然会碰到许多非常challenging的问题，例如low-resolution image classification，频繁新增类别的visual classification等等。本文旨在介绍Visual Classification领域一些我认为比较insightful的paper，以及笔者在实际工作中积累的一些思考。


## [Unsupervised deep feature transfer for low resolution image classification](http://openaccess.thecvf.com/content_ICCVW_2019/papers/RLQ/Wu_Unsupervised_Deep_Feature_Transfer_for_Low_Resolution_Image_Classification_ICCVW_2019_paper.pdf)
本文idea非常简单，作者先用t-SNE算法对high-resolution categories和low-resolution categories的feature进行可视化后发现，那些在HR非常separable的samples，在LR缺难以分开，因此效果非常差。为了解决LR image classification问题，作者提出了这样一个方法来利用HR images信息来辅助LR visual classification。算法详情如下：
1. 用pretrained CNN作为feature extractor，同时提取LR和HR的特征；其中HR label已知，LR label未知
2. 对HR做KMeans聚类，这样就得到了$k$个pseudo label，然后通过比对LR sample与$k$个HR centroid的距离来为LR sample分配pseudo label
3. 对HR的pseudo label与groundtruth，用Feature Transfer Network来优化classification loss
4. 提取LR image feature，过Feature Transfer Network，然后SVM训练之

![UDFT](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-classification/udft.png)

Idea很简单，记录一下看过的一些LR Visual Classification领域常用的从HR生成LR images的方法：将原图resize到$224\times 224$ by bicubic interpolation，然后下采样到$32\times 32$，再resize到$224\times 224$。

作者在VOC2007做了实验，发现本文提出的方法比LR-baseline提升了2%的分类mAP，但相比HR-baseline差距还是很大，这启示我们**高质量数据才是关键，用各种tricky的算法只是尽可能地接近这个上限**。


## Reference
1. He T, Zhang Z, Zhang H, et al. [Bag of tricks for image classification with convolutional neural networks](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 558-567.
2. Wu Y, Zhang Z, Wang G. [Unsupervised deep feature transfer for low resolution image classification](http://openaccess.thecvf.com/content_ICCVW_2019/papers/RLQ/Wu_Unsupervised_Deep_Feature_Transfer_for_Low_Resolution_Image_Classification_ICCVW_2019_paper.pdf)[C]//Proceedings of the IEEE International Conference on Computer Vision Workshops. 2019: 0-0.