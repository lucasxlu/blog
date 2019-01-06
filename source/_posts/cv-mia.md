---
title: "[CV] Medical Image Analysis"
date: 2019-01-06 23:23:36
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

简而言之，尽管我们观念上可能认为ImageNet中的natural images data distribution(例如semantic meaning, image resolution等等)和医学图像的data distribution差距很大，fine-tune似乎不可行，但是作者做了3类实验(classification/detection/segmentation)，发现```从ImageNet上pretrain的模型到医学图像分析认为上进行fine-tune是可行的，而且会带来性能提升，收敛也更快```。

## Deep Learning for Skin Cancer Classification
> Paper: [Dermatologist-level classification of skin cancer with deep neural networks](https://www.nature.com/articles/nature21056.epdf?author_access_token=8oxIcYWf5UNrNpHsUHd2StRgN0jAjWel9jnR3ZoTv0NXpMHRAJy8Qn10ys2O4tuPakXos4UhQAFZ750CsBNMMsISFHIKinKDMKjShCpHIlYPYUHhNzkn6pSnOCt0Ftf6)





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