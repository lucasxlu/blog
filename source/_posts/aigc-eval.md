---
title: "[AIGC] Evaluation"
date: 2024-06-22 14:15:07
mathjax: true
tags:
- Machine Learning
- Deep Learning
- AIGC
- Diffusion Model
catagories:
- Machine Learning
- Deep Learning
- AIGC
- Diffusion Model
---
## Introduction
AIGC 在各行业得到了广泛的应用，相应的算法改进也层出不穷。本文重点不在于梳理具体的生成式模型，而在于如何对 AIGC 模型效果进行合理、科学的评价。

## Methods
### Inception Score
> Paper: [A note on the inception score](https://arxiv.org/pdf/1801.01973)
This metric is motivated by demonstrating that it prefers models that generate realistic and varied images and is correlated with visual quality. The `Inception Score` is a metric for automatically evaluating the quality of image generative models. This metric was shown to correlate well with human scoring of the realism of generated images from the CIFAR-10 dataset. The `IS` uses an Inception v3 Network pre-trained on ImageNet and calculates a statistic of the network's outputs when applied to generated images.

$$
IS(G)=exp(\mathbb{E}_{x\sim p_g}D_{KL}(p(y|x)||p(y)))
$$

where $x\sim p_g$ indicates that x is an image sampled from $p_g$, $D_{KL}(p||q)$ is the KL-divergence between the distributions $p$ and $q$, $p(y|x)$ is the conditional class distribution, and $p(y) = \int_x p(y|x)p_g(x)$ is the marginal class distribution. The exp in the expression is there to make the values easier to compare, so it will be ignored and we will use $ln(IS(G))$ without loss of generality.

In the general setting, the problems with the Inception Score fall into two categories:
1. Suboptimalities of the Inception Score itself 
   * Inception Score is sensitive to small changes in network weights that do not affect the final classification accuracy of the network
   * score calculation and exponentiation
2. Problems with the popular usage of the Inception Score

### CLIPScore
> Paper: [Clipscore: A reference-free evaluation metric for image captioning](https://arxiv.org/pdf/2104.08718)



## Reference
1. Barratt, Shane, and Rishi Sharma. "[A note on the inception score](https://arxiv.org/pdf/1801.01973)." arXiv preprint arXiv:1801.01973 (2018).
2. Wang, Jianyi, Kelvin CK Chan, and Chen Change Loy. "[Exploring clip for assessing the look and feel of images](https://ojs.aaai.org/index.php/AAAI/article/view/25353/25125)." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 2. 2023.
3. Hessel, Jack, et al. "[Clipscore: A reference-free evaluation metric for image captioning](https://arxiv.org/pdf/2104.08718)." arXiv preprint arXiv:2104.08718 (2021).