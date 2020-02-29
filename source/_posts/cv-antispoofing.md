---
title: "[CV] Face Anti-Spoofing"
date: 2018-10-30 00:23:26
mathjax: true
tags:
- Machine Learning
- Deep Learning
- Computer Vision
- Face Anti-Spoofing
catagories:
- Machine Learning
- Deep Learning
- Computer Vision
- Face Anti-Spoofing
---
## Introduction
Face Anti-spoofing，即人脸活体检测，随着iPhoneX FaceID的应用，人脸解锁得到了越来越多的关注，而anti-spoofing无疑是整个人脸解锁环节中非常重要的一环。试想一下，如果连真假脸都区分不出，那安全性无疑是会大打折扣。Face Anti-spoofing在近些年的顶会上也有相关的文献发表。但和众多research benchmark存在的问题一样，目前得dataset capacity太小了，往往很多时候各种model都是在相关数据集上overfitting，更无从谈起实际应用场景了。工业界，因数据量级很大，很多时候也是将其视为一个传统的Binary Classification问题，而根据我们组的模型上线情况反馈来看，Precision和Recall一般都可以到达99.9%+，所以工业界很多时候都是一个数据问题，而非模型和Loss问题。

目前主流的Attack方式有以下3种：
* print attack: 即打印人脸照片攻击
* replay attack: 即播放视频攻击
* 3D mask attack: 即带上3D面具进行攻击

目前主流的Anti-spoofing方法主要有以下几种：
* Image Quality Analysis: 这个很容易理解，因为大多数攻击照片都是拍摄屏幕获得，所以往往会存在一些颜色失真、反光、模糊、形变(recapture时不同角度造成的)、moire pattern (可以由LBP descriptor表示)、边框等，所以这些pattern是很容易被deep models学到，且泛化能力也都不错。有Paper[1]表明，从R Channel中提取的特征比G、B、GrayScale表示能力要更强。
* Command Motion: 就是根据系统发出的指令，用户根据指令进行“眨眼”、“点头”、“转向”、“念一段文字”、“做出某个指定表情”等等来验证活体。
* 3D Depth Information: 真假脸最显著的区别就是活体是立体的，而print/replay attack往往是2D的，所以很容易通过3D深度信息进行区分。此外，在设置Reject Option的时候，也要考虑拍摄距离，不能太远，也不能太近。


本文旨在对CVPR/ECCV/TIP/TIFS等顶会/顶刊Paper的idea做一下梳理。

## Reference
1. Patel K, Han H, Jain A K. [Secure face unlock: Spoof detection on smartphones](http://www.jdl.ac.cn/doc/2011/201711222512198092_hanhu-journal.pdf)[J]. IEEE transactions on information forensics and security, 2016, 11(10): 2268-2283.