---
title: "[CV] Object Detection"
date: 2018-08-28 11:20:05
mathjax: true
tags:
- Machine Learning
- Deep Learning
- Computer Vision
- Object Detection
catagories:
- Machine Learning
- Deep Learning
- Computer Vision
- Object Detection
---
## Introduction
Object Detection是Computer Vision领域一个非常火热的研究方向。并且在工业界也有着十分广泛的应用(例如人脸检测、无人驾驶的行人/车辆检测等等)。本质旨在梳理RCNN--SPPNet--Fast RCNN--Faster RCNN--FCN--Mask RCNN，YOLO v1/2/3, SSD等Object Detection这些非常经典的工作。

> [@LucasX](https://www.zhihu.com/people/xulu-0620/activities)注：本文长期更新。

## RCNN (Region-based CNN)
> Paper: [Rich feature hierarchies for accurate object detection and semantic segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)

### What is RCNN?
这篇Paper可以看做是Deep Learning在Object Detection大获成功的开端，对Machine Learning/Pattern Recognition熟悉的读者应该都知道，**Feature Matters in approximately every task!** 而RCNN性能提升最大的因素之一便是很好地利用了CNN提取的Feature，而不是像先前的detector那样使用手工设计的feature(例如SIFT/LBP/HOG等)。

RCNN可以认为是Regions with CNN features，即(1)先利用[Selective Search算法](https://staff.fnwi.uva.nl/th.gevers/pub/GeversIJCV2013.pdf)生成大约2000个Region Proposal，(2)Pretrained CNN从这些Region Proposal中提取deep feature(from pool5)，(3)然后再利用linear SVM进行one-VS-rest分类。从而将Object Detection问题转化为一个Classification问题，对于Selective Search框选不准的bbox，后面使用<font color="orange">Bounding Box Regression</font>(下面会详细介绍)进行校准。这便是RCNN的主要idea。

![RCNN](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-detection/rcnn.png)

### Details of RCNN
#### Pretraining and Fine-tuning
RCNN先将Base Network在ImageNet上train一个1000 class的classifier，然后将output neuron设置为21(20个Foreground + 1个Background)在target datasets上fine-tune。其中，将由Selective Search生成的Region Proposal与groundtruth Bbox的$IOU\geq 0.5$的定义为positive sample，其他的定义为negative sample。

作者在实验中发现，pool5 features learned from ImageNet足够general，并且在domain-specific datasets上学习non-linear classifier可以获得非常大的性能提升。

#### Bounding Box Regression
输入是$N$个training pairs，$\{(P^i,G^i)\}_{i=1,2,\cdots,N}, P^i=(P_x^i,P_y^i,P_w^i,P_h^i)$代表$P^i$像素点的$(x,y)$坐标点、width和height。$G=(G_x,G_y,G_w,G_h)$代表groundtruth bbox。BBox Regression的目的就是为了学习一种mapping使得proposed box $P$ 映射到 groundtruth box $G$。

将$x,y$的transformation设为$d_x(P),d_y(P)$，属于<font color="red">scale-invariant translation。$w,h$是log-space translation</font>。学习完成后，可将input proposal转换为predicted groundtruth box $\hat{G}$:
$$
\hat{G}_x=P_w d_x(P)+P_x
$$

$$
\hat{G}_y=P_h d_x(P)+P_y
$$

$$
\hat{G}_w=P_w exp(d_w(P))
$$

$$
\hat{G}_h=P_h exp(d_h(P))
$$

每个$d_{\star}(P)$都用一个线性函数来进行建模，使用$pool_5$ feature，权重的学习则使用OLS优化即可：
$$
w_{\star}=\mathop{argmin} \limits_{\hat{w}_{\star}} \sum_i^N (t_{\star}^i-\hat{w}_{\star}^T \phi_5 (P^i))^2 + \lambda||\hat{w}_{\star}||^2
$$

The regression targets $t_{\star}$ for the training pair $(P, G)$ are defined as:
$$
t_x=\frac{G_x-P_x}{P_w}
$$

$$
t_y=\frac{G_y-P_y}{P_h}
$$

$$
t_w=log(\frac{G_w}{P_w})
$$

$$
t_h=log(\frac{G_h}{P_h})
$$

在选取Proposed Bbox的时候，我们只选取离Groundtruth Bbox比较近的($IOU\geq 0.6$)来做Bounding Box Regression。

以上就是Deep Learning在Object Detection领域一个开创性的工作--RCNN。若有疑问，欢迎给我留言！


## SPPNet
> Paper: [Spatial pyramid pooling in deep convolutional networks for visual recognition.](https://arxiv.org/pdf/1406.4729v4.pdf)

### What is SPPNet?
SPPNet(Spatial Pyramid Pooling)是基于RCNN进行改进的一个Object Detection算法。介绍SPPNet之前，我们不妨先来看一下RCNN有什么问题？RCNN，即Region-based CNN，它需要CNN作为base network去做特征提取，而传统CNN需要固定的squared input，而为了满足这个条件，就需要手工地对原图进行裁剪、变形等操作，而这样势必会丢失信息。作者意识到这种现象的原因不在于卷积层，而在于FC Layers需要固定的输入尺寸，因此通过在feature map的SSPlayer可以满足对多尺度的feature map裁剪，从而concatenate得到固定尺寸的特征输入。取得了很好的效果，在detection任务上，region proposal直接在feature map上生成，而不是在原图上生成，因此可以仅仅通过一次特征提取，而不需要像RCNN那样提取2000次(2000个 Region Proposal)，这大大加速了检测效率。

> [@LucasX](https://www.zhihu.com/people/xulu-0620/activities)注：现如今的Deep Architecture比较多采用Fully Convolutional Architecture(全卷积结构)，而不含Fully Connected Layers，在最后做分类或回归任务时，采用Global Average Pooling即可。

![Crop/Warp VS SPP](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-detection/cw_vs_spp.jpg)


### Why SPPNet?
SPPNet究竟有什么过人之处得到了Kaiming He大神的赏识呢？
1. SPP可以在不顾input size的情况下获取fixed size output，这是sliding window做不到的。
2. SPP uses multi-level spatial bins，而sliding window仅仅使用single window size。<font color="red">multi-level pooling对object deformation则十分地robust</font>。
3. SPP can pool features extracted at variable scales thanks to the flexibility of input scales.
4. Training with variable-size images increases scale-invariance and reduces over-fitting.

### Details of SPPNet
#### SPP Layer
SPP Layer can maintain spatial information by pooling in local spatial bins. <font color="red">These spatial bins have sizes proportional to the image size, so the number of bins is fixed regardless of the image size.</font> This is in contrast to the sliding window pooling of the previous deep networks,where the number of sliding windows depends on the input size.

![SPP Layer](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-detection/spp_layer.jpg)

这样，通过不同spatial bin pooling得到的k个 M-dimensional feature concatenation，我们就可以得到fixed length的feature vector了，接下来是不是就可以愉快地用FC Layers/SVM等ML算法train了？

#### SPP for Detection
RCNN需要从2K个Region Proposal中feedforwad Pretrained CNN去提取特征，这显然是非常低效的。SPPNet直接将整张image(possible multi-scale))作为输入，这样就可以只feedforwad一次CNN。然后在<font color="red">feature map层面</font>获取candidate window，SPP Layer pool到fixed-length feature representation of the window。

![Pooling](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-detection/pooling.jpg)

Region Proposal生成阶段和RCNN比较相似，依然是[Selective Search](https://staff.fnwi.uva.nl/th.gevers/pub/GeversIJCV2013.pdf)生成2000个bbox candidate，然后将原始image resize使得$min(w, h)=s$，文中采用 4-level spatial pyramid ($1\times 1, 2\times 2, 3\times 3,6\times 6$, totally 50 bins) to pool the features。对于每个window，该Pooling操作得到一个12800-Dimensional (256×50) 的向量。这个向量作为FC Layers的输入，然后和RCNN一样训练linear SVM去做分类。

训练SPP Detector时，正负样本的采样是基于groundtruth bbox为基准，$IOU\geq 0.3$为positive sample，反之为negative sample。


## Fast RCNN
> Paper: [Fast RCNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)

Fast RCNN是Object Detection领域一个非常经典的算法。它的novelty在于引入了两个branch来做multi-task learning(category classification和bbox regression)。

### Why Fast RCNN?
按照惯例，我们不妨先来看一看之前的算法(RCNN/SPPNet)有什么缺点？
1. 它们(RCNN/SPP)的训练都属于multi-stage pipeline，即先要利用Selective Search生成2K个Region Proposal，然后用log loss去fine-tune一个deep CNN，用Deep CNN抽取的feature去拟合linear SVM，最后再去做Bounding Box Regression。
2. 训练很费时，CNN需要从每一个Region Proposal抽取deep feature来拟合linear SVM。
3. testing的时候慢啊，还是太慢了。因为需要将Deep CNN抽取的feature先缓存到磁盘，再读取feature来拟合linear SVM，你说麻烦不麻烦。

那我们再来看看Fast RCNN为什么优秀？
1. 设计了一个multi-task loss，来同时优化object classification和bounding box regression。
2. Training is single stage.
3. Higher performance than RCNN and SPPNet.

### Details of Fast RCNN
![Fast RCNN](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-detection/fastrcnn.jpg)

Fast RCNN pipeline如上图所示：它将whole image with several object proposals作为输入，CNN抽取feature，对于每一个object proposal，<font color="red">region of interest (RoI) pooling layer extracts a fixed-length feature vector from the feature map</font>，然后将走过RoI Pooling Layer的feature vector输送到随后的multi-branch，一同做classification和bbox regression。

可以看到，Fast RCNN模型里面一个非常重要的组件叫做<font color="red">RoI Pooling</font>，那么接下来我们就来细细分析一下RoI Pooling究竟是何方神圣。

#### The RoI pooling layer
Fast RCNN原文里是这样说的：
> RoI pooling layer uses max pooling to convert the features inside any valid region of interest into a small feature map with a fixed spatial extent of $H\times W$ (e.g., $7\times 7$), where $H$ and $W$ are layer hyper-parameters that are independent of any particular RoI.
> 
> In this paper, an RoI is a rectangular window into a conv feature map. Each RoI is defined by a four-tuple $(r, c, h, w)$ that specifies its top-left corner $(r, c)$ and its height and width $(h, w)$.
>
> RoI max pooling works by dividing the $h\times w$ RoI window into an $H\times W$ grid of sub-windows of approximate size $h/H \times w/W$ and then max-pooling the values in each sub-window into the corresponding output grid cell. Pooling is applied independently to each feature map channel, as in standard max pooling. The RoI layer is simply the special-case of the spatial pyramid pooling layer used in SPPnets [11] in which there is only one pyramid level. We use the pooling sub-window calculation given in [11].

什么意思呢？就是在任何valid region proposal里面，把某层的feature map划分成多个小方块，每个小方块做max pooling，这样就得到了尺寸更小的feature map。

#### Fine-tuning for detection
##### Multi-Task Loss
之前也说过，Fast RCNN同时做了$K+1$ (K个object class + 1个background) 类的classification($p=(p_0,p_1,\cdots,p_K)$)和bbox regression($t^k = (t^k_x, t^k_y, t^k_w, t^k_h)$)。

We use the parameterization for $t^k$ given in [9], in which <font color="red">$t^k$ specifies a scale-invariant translation and log-space height/width shift relative to an object proposal</font>(对linear regression熟悉的读者不妨思考一下为什么要对width和height做log). Each training RoI is labeled with a ground-truth class $u$ and a ground-truth bounding-box regression target $v$. We use a multi-task loss $L$ on each labeled RoI to jointly train for classification and bounding-box regression:
$$
L(p,u,t^u,v)=L_{cls}(p,u)+\lambda [u\geq1]L_{loc}(t^u,v)
$$
$L_{cls}(p,u)=-logp_u$ is log loss for true class $u$.

我们再来看看Loss Function的第二部分(即regression loss)，$[u\geq 1]$代表只有满足$u\geq 1$时这个式子才为1，否则为0。在我们的setting中，background的$[u\geq 1]$自然而然就设为0啦。我们接着分析regression loss，既然是regression，惯常的手法是使用MSE Loss对不对？但是MSE Loss属于Cost-sensitive Loss啊，对outliers非常的敏感，因此Ross大神使用了更加柔和的$Smooth L_1 Loss$。
$$
L_{loc}(t^u,v)\sum_{i\in \{x,y,w,h\}} smooth_{L_1}(t_i^u-v_i)
$$

Smooth L1 Loss写得详细一点呢，就是这样的：
$$
smooth_{L_1}(x)=
\begin{cases}
0.5x^2 & if |x|<1\\
|x|-0.5 & otherwise
\end{cases}
$$

> [@LucasX](https://www.zhihu.com/people/xulu-0620/activities)注：想详细了解Machine Learning中的Loss，请参考我的[另一篇文章](https://lucasxlu.github.io/blog/2018/07/24/ml-loss/)。

##### Mini-batch sampling
在Fine-tuning阶段，每个mini-batch随机采样自$N=2$类image，每一类都是64个sample，与groundtruth bbox $IOU\geq 0.5$的设为foreground samples[$u=1$]，反之为background samples[$u=0$]。

### Fast R-CNN detection
因为Fully Connected Layers的计算太费时，而FC Layers的计算显然就是大矩阵相乘，因此很容易联想到用truncated SVD来进行加速。

In this technique, a layer parameterized by the $u\times v$ weight matrix $W$ is approximately factorized as:
$$
W\approx U\Sigma_t V^T
$$
$U$是由$W$前$t$个left-singular vectors组成的$u\times t$矩阵，$\Sigma_t$是包含$W$矩阵前$t$个singular value的$t\times t$对角矩阵，$V$是由$W$前$t$个right-singular vectors组成的$v\times t$矩阵。Truncated SVD可以将参数从$uv$降到$t(u+v)$。

这里值得一提的有两点：
1. $conv_1$ layers feature map通常都足够地general，并且task-independent。因此可以直接用来抽feature就行。
2. Region Proposal的数量对Detection的性能并没有什么太大影响 (关键还是看feature啊！以及sampling mini-batch的时候正负样本的不均衡问题，详情请参阅kaiming He大神的[Focal Loss](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf))。


## Faster RCNN
> Paper: [Faster r-cnn: Towards real-time object detection with region proposal networks](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)

Faster RCNN也是Object Detection领域里一个非常具有代表性的工作，一个最大的改进就是Region Proposal Network(RPN)，RPN究竟神奇在什么地方呢？我们先来回顾一下RCNN--SPP--Fast RCNN，这些都属于two-stage detector，什么意思呢？就是先利用<font color="red">Selective Search</font>生成2000个Region Proposal，然后再将其转化为一个机器学习中的分类问题。而<font color="red">Selective Search</font>实际上是非常低效的，RPN则很好地完善了这一点，即直接从整个Network Architecture里生成Region Proposal。RPN是一种全卷积网络，它可以同时预测object bounds，以及objectness score。因为RPN的Feature是和Detection Network共享的，所以整个Region Proposal的生成几乎是cost-free的。所以，这也就是Faster RCNN中**Faster**一词的由来。


### What is Faster RCNN?
$$
Faster RCNN = Fast RCNN + RPN
$$
按照惯例，一个算法的提出显然是为了解决之前算法的不足。那之前的算法都有什么问题呢？
如果对之前的detector熟悉的话，shared features between proposals已经被解决，但是<font color="red">Region Proposal的生成变成了最大的计算瓶颈</font>。这便是RPN产生的缘由。

作者注意到，conv feature maps used by region-based detectors也可以被用于生成region proposals。在这些conv features顶端，<font color="red">通过添加两个额外的卷积层来构造RPN：一个conv layer用于encode每个conv feature map position到一个低维向量(256-d)；另一个conv layer在每一个conv feature map position中输出k个region proposal with various scales and aspect ratios的objectness score和regression bounds。</font>下面重点介绍一下RPN。

### Region Proposal Network
RPN是一个全卷积网络，可以接受任意尺寸的image作为输入，并且输出一系列object proposals以及其对应的objectness score。那么RPN是如何生成region proposals的呢？

首先，在最后一个shared conv feature map上slide一个小网络，这个小网络全连接到input conv feature map上$n\times n$的spatial window。而每一个sliding window映射到一个256-d的feature vector，这个feature vector输入到两个fully connected layers--一个做box regression，另一个做box classification。值得注意的是，因为这个小网络是以sliding window的方式操作的，所以fully-connected layers在所有的spatial locations都是共享的。该结构由一个$n\times n$ conv layer followed by two sibling $1\times 1$ conv layers组成。

#### Translation-Invariant Anchors
对于每个sliding window location，同时预测$k$个region proposals，所以regression layer有$4k$个encoding了$k$个bbox坐标的outputs。classification layer有$2k$个scores(每个region proposal估计object/non-object的概率)。

> The $k$ proposals are parameterized relative to $k$ reference boxes, called anchors. Each anchor is centered at the sliding window in question, and is associated with a scale and aspect ratio.

文章中采用3个scales和3个aspect ratios，所以每个sliding position一共得到$k=9$个anchors。所以对于每个$W\times H$的conv feature map一共产生$WHk$个anchor。这种方法一个很重要的性质就是**它是translation invariant**。

#### A Loss Function for Learning Region Proposals
训练RPN时，我们为每一个anchor分配binary class(即是不是一个object)。我们为以下这两类anchors分配positive label:
1. 与groundtruth bbox有最高IoU的anchor
2. 与任意groundtruth bbox $IoU\geq 0.7$的anchor

同时，可能出现一个groundtruth bbox分配到多个anchor的情况，我们将与所有groundtruth bbox $IoU\leq 0.3$的anchor设为negative anchor。而那些既不是positive又不是negative的anchor则对training过程没有用处。然后，Faster RCNN的Loss Function就变成：
$$
L(\{p_i\},\{t_i\})=\frac{1}{N_{cls}} \sum_i L_{cls}(p_i,p_i^{\star}) + \lambda \frac{1}{N_{reg}} \sum_i p_i^{\star} L_{reg}(t_i,t_i^{\star})
$$
$p_i$是anchor $i$ 被预测为是一个object的概率，若anchor为positive，则groundtruth label $p_i^{{\star}}$为1；若anchor为negative则为0；$t_i$是包含4个预测bbox坐标点的向量，$t_i^{\star}$是groundtruth positive anchor坐标点的向量。$L_{cls}$是二分类的Log Loss(object VS non-object)。对于regression loss，文章使用$L_{reg}(t_i,t_i^{\star})=R(t_i-t_i^{\star})$，其中$R$是Smooth L1 Loss(和Fast RCNN中一样)。$p_i^{\star} L_{reg}$表示<font color="red">仅仅在positive anchor ($p_i^{\star}=1$)时才被激活，否则($p_i^{\star}=0$)不激活</font>。

Bounding Box Regression依旧是采用之前的pipeline：
$$
t_x=\frac{x-x_a}{w_a},t_y=\frac{y-y_a}{h_a},t_w=log(\frac{w}{w_a}),t_h=log(\frac{h}{h_a})
$$

$$
t_x^{\star}=\frac{x^{\star}-x_a}{w_a},t_y^{\star}=\frac{y^{\star}-y_a}{h_a},t_w^{\star}=log(\frac{w^{\star}}{w_a}),t_h^{\star}=log(\frac{h^{\star}}{h_a})
$$

> In our formulation, the features used for regression are of the same spatial size $(n\times n)$ on the feature maps. **To account for varying sizes, a set of $k$ bounding-box regressors are learned. Each regressor is responsible for one scale and one aspect ratio, and the $k$ regressors do not share weights**. As such, it is still possible to predict boxes of various sizes even though the features are of a fixed size/scale.

#### Sharing Convolutional Features for Region Proposal and Object Detection
1. Fine-tune在ImageNet上Pretrain的RPN来完成region proposal task。
2. 利用RPN生成的region proposal来train Fast RCNN。注意在这一步骤中RPN和Faster RCNN没有共享卷积层。
3. 利用detector network来初始化RPN训练，但是我们fix shared conv layers，仅仅fine-tune单独属于RPN的层。注意在这一步骤中RPN和Fast RCNN共享了卷积层。
4. Fix所有shared conv layers，fine-tune Fast RCNN的fc layers。至此，RPN和Fast RCNN共享卷积层，并且形成了一个unified network。



## Reference
1. Girshick, Ross, et al. ["Rich feature hierarchies for accurate object detection and semantic segmentation."](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.
2. He, Kaiming, et al. ["Spatial pyramid pooling in deep convolutional networks for visual recognition."](https://arxiv.org/pdf/1406.4729v4.pdf) European conference on computer vision. Springer, Cham, 2014.
3. Girshick, Ross. ["Fast r-cnn."](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf) Proceedings of the IEEE international conference on computer vision. 2015.
4. Ross, Tsung-Yi Lin Priya Goyal, and Girshick Kaiming He Piotr Dollár. ["Focal Loss for Dense Object Detection."](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)
5. Ren, Shaoqing, et al. ["Faster r-cnn: Towards real-time object detection with region proposal networks."](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf) Advances in neural information processing systems. 2015.