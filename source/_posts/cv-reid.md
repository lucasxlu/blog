---
title: "[CV] Re-identification"
date: 2020-03-05 11:34:36
mathjax: true
tags:
- Computer Vision
- Machine Learning
- Deep Learning
- Visual Search
- Re-identification
catagories:
- Computer Vision
- Machine Learning
- Deep Learning
- Visual Search
- Re-identification
---
## Introduction
ReID可以看作是Visual Search的一个specific task，也在工业界有着非常广泛的应用场景。本文主要记录一些ReID领域比较insightful的paper。按照惯例，介绍某个新领域之前，首先得需要了解该领域的benchmark datasets以及performance evaluation metric。在ReID领域，主要有**CMC**、**mAP**和**rank1 Accuracy**作为evaluation metric。

* CMC
The CMC curve shows the probability that a query identity appears in different-sized candidate lists. This evaluation measurement is valid only if there is only one ground truth match for a given query. In this case, precision and recall are the same issue. However, if multiple ground truths exist, the CMC curve is biased because "recall" is not considered.

* mAP
For each query, we calculate the area under the Precision-Recall curve, which is known as average precision (AP). Then, the mean value of APs of all queries, i.e., mAP, is calculated, which considers both precision and recall of an algorithm, thus providing a more comprehensive evaluation. When average precision (AP) is used, rank lists in Fig. 3(b) and Fig. 3(c) are effectively distinguished.

下图直观展示了CMC和mAP的区别：

![CMC VS mAP](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-reid/cmc_map.png)

* rank-N Accuracy
给定一个ID为m1的query probe，从gallery里返回M个retrieval list的前N个召回的retrieval结果里至少有一次hit包含ID为m1。

Person ReID常用的有以下几个数据集：
* [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)
In our dataset, if the area ratio is larger than 50%, the DPM bbox is marked as "good" (a routine in object detection); if the ratio is smaller than 20%, the DPM bbox is marked as "distractor"; otherwise, the bbox is marked as "junk", meaning that this image is of zero influence to re-id accuracy.


## Reference
1. Zheng L, Shen L, Tian L, et al. [Scalable person re-identification: A benchmark](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)[C]//Proceedings of the IEEE international conference on computer vision. 2015: 1116-1124.