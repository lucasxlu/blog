---
title: "[CV] Medical Image Analysis"
date: 2019-01-12 22:37:36
mathjax: true
tags:
- Computer Vision
- Machine Learning
- Deep Learning
- Medical Image Analysis
catagories:
- Computer Vision
- Machine Learning
- Deep Learning
- Medical Image Analysis
---
## Introduction
医疗AI是现如今工业界和学术界都非常火热的方向，也是AI落地非常有价值的方向。因毕业论文需要涉及Medical Image Analysis相关的方向，所以本文旨在收集并梳理Deep Learning在Medical Image Analysis领域一些具有代表性的Paper以及Report。

> [@LucasX](https://www.zhihu.com/people/xulu-0620/activities)注：本文长期更新。

## CNN for Medical Image Analysis. Full Training or Fine Tuning?
> Paper: [Convolutional neural networks for medical image analysis: Full training or fine tuning?](https://arxiv.org/pdf/1706.00712.pdf)

Our experiments consistently demonstrated that 
1. the use of a pre-trained CNN with adequate fine-tuning outperformed or, in the worst case, performed as well as a CNN trained from scratch;
2. fine-tuned CNNs were more robust to the size of training sets than CNNs trained from scratch; 
3. neither shallow tuning nor deep tuning was the optimal choice for a particular application; and 
4. our layer-wise fine-tuning scheme could offer a practical way to reach the best performance for the application at hand based on the amount of available data.

简而言之，尽管我们观念上可能认为ImageNet中的natural images data distribution(例如semantic meaning, image resolution等等)和医学图像的data distribution差距很大，fine-tune似乎不可行，但是作者做了3类实验(classification/detection/segmentation)，发现**从ImageNet上pretrain的模型到医学图像分析认为上进行fine-tune是可行的，而且会带来性能提升，收敛也更快**。

## Deep Learning for Skin Cancer Classification
> Paper: [Dermatologist-level classification of skin cancer with deep neural networks](https://www.nature.com/articles/nature21056.epdf?author_access_token=8oxIcYWf5UNrNpHsUHd2StRgN0jAjWel9jnR3ZoTv0NXpMHRAJy8Qn10ys2O4tuPakXos4UhQAFZ750CsBNMMsISFHIKinKDMKjShCpHIlYPYUHhNzkn6pSnOCt0Ftf6)

[Inception v3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf) (pretrained on ImageNet)用于做skin cancer classification，算法效果超过了人类医生。模型没什么太大的新意，记一下医学图像领域常用的两个metric：
$$
sensitivity=\frac{True Positive}{Positive}
$$

$$
specificity=\frac{True Negative}{Negative}
$$

## Deep Learning for Chest X-rays Recognition
> Paper: [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning](https://arxiv.org/pdf/1711.05225v3.pdf)

Deep Model尽管性能超群，但是Interpretability却非常差，尤其是AI在医学、金融等领域的应用，就会非常看重Interpretability(所以好多金融系统依然还在用规则，而非Machine Learning)。CheXNet是Andrew Ng团队的成果，基础idea是DenseNet做classification，[CAM](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)做可视化，来显示哪些区域是最discriminative的。



## Reference
1. Ding, Yiming, et al. ["A deep learning model to predict a diagnosis of Alzheimer disease by using 18F-FDG PET of the brain."](https://pubs.rsna.org/doi/pdf/10.1148/radiol.2018180958) Radiology (2018): 180958.
2. Tajbakhsh, Nima, et al. ["Convolutional neural networks for medical image analysis: Full training or fine tuning?."](https://arxiv.org/pdf/1706.00712.pdf) IEEE transactions on medical imaging 35.5 (2016): 1299-1312.
3. De Fauw, Jeffrey, et al. ["Clinically applicable deep learning for diagnosis and referral in retinal disease."](https://www.nature.com/articles/s41591-018-0107-6) Nature medicine 24.9 (2018): 1342. 
4. Kermany, Daniel S., et al. ["Identifying medical diagnoses and treatable diseases by image-based deep learning."](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5?code=cell-site) Cell 172.5 (2018): 1122-1131.
5. Gulshan, Varun, et al. ["Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs."](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45732.pdf) Jama 316.22 (2016): 2402-2410.
6. Rajpurkar, Pranav, et al. ["Deep learning for chest radiograph diagnosis: A retrospective comparison of the CheXNeXt algorithm to practicing radiologists."](https://journals.plos.org/plosmedicine/article/file?id=10.1371/journal.pmed.1002686&type=printable) PLOS Medicine 15.11 (2018): e1002686.
7. [Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet](https://journals.plos.org/plosmedicine/article/file?id=10.1371/journal.pmed.1002699&type=printable) Bien N, Rajpurkar P, Ball RL, Irvin J, Park A, et al. (2018) Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet. PLOS Medicine 15(11): e1002699. https://doi.org/10.1371/journal.pmed.1002699
8. [Deep learning for chest radiograph diagnosis: A retrospective comparison of the CheXNeXt algorithm to practicing radiologists](https://journals.plos.org/plosmedicine/article/file?id=10.1371/journal.pmed.1002686&type=printable) Rajpurkar P, Irvin J, Ball RL, Zhu K, Yang B, et al. (2018) Deep learning for chest radiograph diagnosis: A retrospective comparison of the CheXNeXt algorithm to practicing radiologists. PLOS Medicine 15(11): e1002686. https://doi.org/10.1371/journal.pmed.1002686
9. Esteva, Andre, et al. ["Dermatologist-level classification of skin cancer with deep neural networks."](https://www.nature.com/articles/nature21056.epdf?author_access_token=8oxIcYWf5UNrNpHsUHd2StRgN0jAjWel9jnR3ZoTv0NXpMHRAJy8Qn10ys2O4tuPakXos4UhQAFZ750CsBNMMsISFHIKinKDMKjShCpHIlYPYUHhNzkn6pSnOCt0Ftf6) Nature 542.7639 (2017): 115.