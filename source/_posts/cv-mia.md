---
title: "[CV] Medical Image Analysis"
date: 2019-01-31 14:34:36
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


## Clinical Skin Lesion Diagnosis using Representations Inspired by Dermatologist Criteria
> Paper: [Clinical skin lesion diagnosis using representations inspired by dermatologist criteria](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Yang_Clinical_Skin_Lesion_CVPR_2018_paper.pdf)

这是一篇发表在[CVPR'18](http://openaccess.thecvf.com/CVPR2018.py)上的文章，主要contribution是设计了一种novel的手工feature来进行skin disease recognition。该手工特征主要涉及Structure, Color, Shape。并且在SD-198 dataset上取得了State-of-the-art的效果。医学图像和我们常规的natural image不同，intra-class diversity非常大，我看了SD-198的图片，DNN实在是非常不好分。
* 对于Structure，由texture distribution来表示；主要通过multiple space的symmetry property来决定。
* 对于Color，通过ColorName来和skin lesion相关的color来表示；以及通过引入每种颜色的continuous value来区分颜色相同、但颜色程度不同的情况。
* 对于Shape，作者利用peripheral symmetry和constrained compactness来进行表示。

### Criteria of Skin Disease
作者通过调研了大量skin lesion的文献，总结了skin diagnosis的ABCD criteria:
* A (Asymmetry): lesion shape, contour, colors and structures.
* B (Border): ill-defined and irregular border of lesion.
* C (Color variegation): skin lesion的color并非统一。
* D (Diameter): skin lesion的大致直径。

### Medical Representations
介绍完了上面的ABCD criteria，下面我们直接来看本文的核心——medical representation。Medical representation可以通过如下方式生成：
* skin lesion的面积、直径(D)、边缘(B)由连通区域的像素点数量，以及主轴决定。
* A可由连通区域的geometric information决定。
* C可通过不同color space共同决定，这样可以保证即使有略微的颜色变化(例如illumination影响)，依然具备discrimination。

下面我们来进行详细的讲解。

#### Structure Representation
##### Multi-Space Texture of Lesion (MST-L)
为了有效地表示skin lesion的structure，我们基于不同的color space来计算texture representation，来减少环境的影响。对于每一张clinical image $x$，我们提出了multi-space texture ($MST(x)$):
$$
MST(x)=[G_i(x)]_{i=1}^K
$$
$G_i(x)$代表从第$i$个color channel抽取的texture feature，$K$代表color space总数。在本文中，作者使用了3个color space，即Hue, Saturation, Brightness；并对每一个color space利用SIFT进行feature extraction。

##### Texture Symmetry of Lesion (TS-L)
skin lesion的不对称性也是个非常discriminative的特征。作者利用MBD+算法进行lesion region检测，然后根据主轴对检测出的区域划分为两个部分——$L(x)_1$和$L(x)_2$。然后对每个部分进行texture feature extraction，最后对第$i$个color space的texture symmetry进行如下表示：
$$
TS_i(x)=[G_i(L(x)_1), G_i(L(x)_2), S_i(x)]
$$
其中，$S_i(x)=\{|g_{ij}^1-g_{ij}^2|\}_{j=1}^d$，$d$代表抽取feature的dimension，$g_{ij}^1$和$g_{ij}^2$代表第$j$个entry $G_i(L(x)_1)$和$G_i(L(x)_2)$。**我们在Hue space中进行texture symmetry度量，因为Hue space在不同light intensity情况下依然是scale-invariant + shift-invariant**。

#### Color Representation
##### Color Name of Lesion (CN-L)
在$L\times a\times b$ space中，我们对每一个color bin计算pprobability vector $P=[p(C_l|c)]_{l=1}^M$：
$$
[p(C_l|c)]_{l=1}^M\propto \sum_i^N p(C_l|c_i)g^{\sigma}(|c_i-c|_{Lab})
$$
$c$代表color bin的original value，$c_i$代表$c$的Lab value，$N=387$代表color bin的总数，$C$代表basic colors set。$g^{\sigma}$代表$\sigma=5$的guassian kernel。作者通过对skin lesion的调研，设置$M=8$，$C=\{red, pink, purple, yellow, white, black, brown, blue\}$。最终lesion的color name $CN(x)$为：
$$
CN(x)=\mathop{argmax} \limits_{C_l}[p(C_l|c)]_{l=1}^M
$$

##### Continuous Color Values of Lesion (CCV-L)
除了CN-L，作者还对每个lesion设置了continuous value来代表color的不同程度。对于每个bin $c$，我们定义continuous color value $CCV(c)$:
$$
CCV(c)\propto p(C,c)\times \theta(c)
$$

其中$p(C,c)$代表将color bin $c$ 映射到其最近color name $C$的概率。$\theta(c)$代表pixel的权重值：
$$
\theta(c)=\sum_{|c|}n(c)u(c)
$$
$n(c)$代表图片中对于color的frequency，$u(c)$代表RGB space中color bin $c$的color value。

#### Shape Representation
对于形状特征，本文主要关注 (1) shape symmetry，(2) lesion的constrained compactness。

##### Peripheral Symmetry of Lesion (PS-L)
同样应用MBD+算法进行lesion region detection，划分等面积的两个部分$L(x)_1$和$L(x)_2$。然后计算两个parts lesion的peripheral symmetry：
$$
PS(x)=F(A(L(x)^1), A(L(x)^2))
$$
其中$A(\cdot)$代表从lesion中抽取的feature，$F(\cdot, \cdot)$代表concatenation operation。

##### Adaptive Compactness of Lesion (AC-L)
对于skin disease region，圆的近似程度对diagnosis非常有帮助。因此可以用如下方式来表示：
$$
Com=\frac{4\pi A}{P^2}
$$
$A$代表面积，$P$代表lesion的周长。

lesion的面积表示：
$$
A_L=\sum_{z\in L(x)}p(C|c,z)
$$
$z$代表lesion $L(x)$的像素，$p(C|c,z)$是在color name feature中将color映射到特定颜色类型的概率，$p(C|c,z)$反映了像素$z$位于lesion center的重要性。

实验结果表明，本文设计的特征比传统方法/DCNN都要好，metric为Accuracy和Sensitivity。



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
10. Codella, Noel CF, et al. ["Deep learning ensembles for melanoma recognition in dermoscopy images."](https://arxiv.org/ftp/arxiv/papers/1610/1610.04662.pdf) IBM Journal of Research and Development 61.4 (2017): 5-1.
11. Haofu, Liao, and Jiebo Luo. ["A Deep Multi-Task Learning Approach to Skin Lesion Classification."](https://www.aaai.org/ocs/index.php/WS/AAAIW17/paper/view/15094/14715) Workshops at the Thirty-First AAAI Conference on Artificial Intelligence. 2017.
12. Yang, Jufeng, et al. ["Clinical skin lesion diagnosis using representations inspired by dermatologist criteria."](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Yang_Clinical_Skin_Lesion_CVPR_2018_paper.pdf) Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
13. Ge, Zongyuan, et al. ["Skin disease recognition using deep saliency features and multimodal learning of dermoscopy and clinical images."](https://link.springer.com/chapter/10.1007/978-3-319-66179-7_29) International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2017.