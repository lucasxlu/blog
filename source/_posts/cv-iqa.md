---
title: "[CV] Image Quality Assessment"
date: 2018-11-04 16:32:11
mathjax: true
tags:
- Computer Vision
- Machine Learning
- Deep Learning
- Digital Image Processing
catagories:
- Computer Vision
- Machine Learning
- Deep Learning
- Digital Image Processing
---
## Introduction
Image Quality Assessment (IQA) 是计算机视觉领域一个非常重要的研究方向，并且在许多方向也有着非常好的落地场景(例如我在滴滴出行实习时，就需要设计算法来实现对网约车司机上传的证件照进行图像质量分析，若存在大规模的反光(reflection)、模糊(blur)等，就需要予以拒绝)；此外，IQA也常常被用于[Face Anti-Spoofing](https://lucasxlu.github.io/blog/2018/10/30/cv-antispoofing/)，因为有时候print/replay attack的图片/视频 和活体相比，其图像质量往往会比较差(例如颜色失真、反光、模糊、变形等)，因此也是一个非常显著的特征。

IQA主要分为3种：(1) 将distorted image和original image进行质量比较的，称为*full reference*。(2) 当reference image不可获取时，称为*no-reference*。(3) 当reference image只有部分可以获取时，称为*reduced reference*。

IQA主要的Metric是*MSE*, *PSNR (Peak Signal-to-Noise Ratio)* 和 *SSMI (structural similarity)*。


> [@LucasX](https://www.zhihu.com/people/xulu-0620/activities)注：本文长期更新。


## Reference
1. Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, ["Image quality assessment: From error visibility to structural similarity,"](http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf) IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, Apr. 2004.
2. Talebi, Hossein, and Peyman Milanfar. ["Nima: Neural image assessment."](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8352823) IEEE Transactions on Image Processing 27.8 (2018): 3998-4011.