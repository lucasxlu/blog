---
title: "[DL] Improving Robustness of Deep Learning Models"
date: 2024-05-22 11:00:00
mathjax: true
tags:
- Machine Learning
- Deep Learning
catagories:
- Algorithm
- Machine Learning
---
# Background
* PyTorch 模型训练时的预处理环节与实车部署环节无法完全对齐，我们回灌 case 发现：PyTorch 模型表现正常，但 fp32 & fp16 的 trt 模型存在线弯折、左右甩等异常现象。模型对上游的一些 perturbation 过于敏感，希望在训练过程中增强鲁棒性。
* 不同初始化&训练方式对模型的 robustness、generality 有较大的影响，需要在训练过程中选择最优的训练方式。

# Methods
## Are All Layers Created Equal?

![Robustness results for FCN 3 × 256 on MNIST](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-robustness/fig1.png)

主要结论：
* DL 模型中，不同 layer 的重要性其实是不同的，low-level layers 在训练过程中更新得更激进，high-level layers 相对缓和一些
* re-initialization 效果优于 re-randomization，paper 里验证了 re-initialization 的方式能提升模型鲁棒性

## Fast Gradient Sign Method
Adversarial training 能够 （1）提高模型应对恶意对抗样本时的鲁棒性；（2）作为一种 regularization，减少 overfitting，提高泛化能力。[FGSM](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html#fast-gradient-sign-attack) 算法 pipeline 如下：
1. Calculate the loss after forward propagation,
2. Calculate the gradient with respect to the pixels of the image,
3. Nudge the pixels of the image ever so slightly in the direction of the calculated gradients that maximize the loss calculated above

## Averaging Weights Leads to Wider Optima and Better Generalization
通过将训练过程中多个 snapshot 的 checkpoint 进行 weight averaging，得到 wider optima (& better generalization)。这个 idea 属于 semi-supervised learning 里常用的 tricks 了，确实很 work。方法非常简单，并且 PyTorch 也集成了 SWA 和 EMA 的实现：

![SWA](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-robustness/swa.png)

此外，也有论文验证了简单[将 SWA 应用在 object detection & segmentation 任务](https://zhuanlan.zhihu.com/p/341190337)中，也能带来提升。

以及...在 quantized training 过程中引入 SWA 也能使得模型对 quantization noise 更加鲁棒。

![SWALP](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-robustness/swalp.png)

## Model Soups
一共提出来 3 种 model soups 方案（uniform soup, greedy soup, learned soup），通过 averaging model weights 就能得到性能提升。其中最有效的是 greedy soup：

![GreedySoup](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-robustness/greedy_soup.png)

## RegMixup

![RegMixup](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-robustness/regmixup.png)

## Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty
以 rotation prediction 作为 pre-text task 进行 self-supervised training，能够很好地提升模型 robustness：

![RotNet](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-robustness/rotnet.png)

## Unsupervised Adversarial Training (UAT)
对抗 adversarial attack 的一个有效方法是增加更多的训练数据，但收集 labeled dataset 毕竟成本比较高。作者提出了一个基于 unsupervised learning 的方法来从海量无标签数据中进行学习，从而增强模型的 robustness。无监督 smooth loss 可以是以下两种形态：

* Unsupervised Adversarial Training with Online Targets (UAT-OT)

![UAT-OT](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-robustness/uat_ot.png)

* Unsupervised Adversarial Training with Fixed Targets (UAT-FT)

![UAT-FT](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-robustness/uat_ft.png)

## Adversarial Robustness is at Odds with Lazy Training
Lazy training 方式训练的 DL 模型容易受到 adversarial attack。
Lazy training：网络在训练过程中的权重变化较小，使得网络在初始化附近表现得像它的线性化。在懒惰训练下，神经网络容易受到对抗性攻击，因为网络在初始化附近的局部线性特性使得对抗性攻击更容易找到梯度上升路径，从而产生错误预测。
提高网络鲁棒性以应对对抗性攻击的方法：
1. 在训练过程中使用对抗训练：通过生成对抗性样本并将其与原始样本一起训练网络，可以提高网络的鲁棒性。
2. 使用更大的网络宽度：增加网络宽度可以提高网络的鲁棒性，但需要注意网络宽度与输入维度之间的平衡。
3. 探索更强的防御方法：研究新的防御方法可能有助于找到更有效的方法来抵御对抗性攻击。

## Robust fine-tuning of zero-shot models
主要分为两个步骤：
* 首先正常 fine-tuning
* 在 weight space 进行 interpolation 操作（和 SWA 其实没啥本质区别...）
这种方法充分利用了零样本模型在分布偏移下的鲁棒性，同时保留了微调模型在目标分布上的高性能。

![WiSE-FT](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dl-robustness/wise_ft.png)

# Reference
1. Zhang, Chiyuan, Samy Bengio, and Yoram Singer. "[Are all layers created equal?](https://www.jmlr.org/papers/volume23/20-069/20-069.pdf)." The Journal of Machine Learning Research 23.1 (2022): 2930-2957.
2. Liu, Mengchen, et al. "[Analyzing the noise robustness of deep neural networks](https://ml.cs.tsinghua.edu.cn/~jun/pub/robust-dnn.pdf)." 2018 IEEE Conference on Visual Analytics Science and Technology (VAST). IEEE, 2018.
3. https://neptune.ai/blog/adversarial-attacks-on-neural-networks-exploring-the-fast-gradient-sign-method
4. [【炼丹技巧】功守道：NLP中的对抗训练 + PyTorch实现](https://zhuanlan.zhihu.com/p/91269728)
5. Izmailov, Pavel, et al. "[Averaging weights leads to wider optima and better generalization](https://arxiv.org/pdf/1803.05407)." arXiv preprint arXiv:1803.05407 (2018).
6. Zhang, Haoyang, et al. "[Swa object detection](https://arxiv.org/pdf/2012.12645)." arXiv preprint arXiv:2012.12645 (2020).
7. Wortsman, Mitchell, et al. "[Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://proceedings.mlr.press/v162/wortsman22a/wortsman22a.pdf)." International Conference on Machine Learning. PMLR, 2022.
8. Yang G, Zhang T, Kirichenko P, et al. [SWALP: Stochastic weight averaging in low precision training](https://proceedings.mlr.press/v97/yang19d/yang19d.pdf)[C]//International Conference on Machine Learning. PMLR, 2019: 7015-7024.
9. Pinto, Francesco, et al. "[Regmixup: Mixup as a regularizer can surprisingly improve accuracy and out distribution robustness](https://arxiv.org/pdf/2206.14502)." arXiv preprint arXiv:2206.14502 (2022).
10. Pinto, Francesco, et al. "[Using mixup as a regularizer can surprisingly improve accuracy & out-of-distribution robustness](https://proceedings.neurips.cc/paper_files/paper/2022/file/5ddcfaad1cb72ce6f1a365e8f1ecf791-Paper-Conference.pdf)." Advances in Neural Information Processing Systems 35 (2022): 14608-14622.
11. Hendrycks, Dan, et al. "[Using self-supervised learning can improve model robustness and uncertainty](https://proceedings.neurips.cc/paper/2019/file/a2b15837edac15df90721968986f7f8e-Paper.pdf)." Advances in neural information processing systems 32 (2019).
12. Alayrac, Jean-Baptiste, et al. "[Are labels required for improving adversarial robustness?](https://proceedings.neurips.cc/paper_files/paper/2019/file/bea6cfd50b4f5e3c735a972cf0eb8450-Paper.pdf)." Advances in Neural Information Processing Systems 32 (2019).
13. Wang, Yunjuan, et al. "[Adversarial robustness is at odds with lazy training](https://proceedings.neurips.cc/paper_files/paper/2022/file/2aab664e0d1656e8b56c74f868e1ea69-Paper-Conference.pdf)." Advances in Neural Information Processing Systems 35 (2022): 6505-6516.
14. Wortsman, Mitchell, et al. "[Robust fine-tuning of zero-shot models](https://openaccess.thecvf.com/content/CVPR2022/papers/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.pdf)." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.