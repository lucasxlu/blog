---
title: "[CV] Object Detection"
date: 2018-11-11 21:07:05
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


## SSD
> Paper: [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325v5.pdf)

SSD是one-stage detector里一个非常著名的算法，那什么叫做one-stage和two-stage呢？回想一下，从DL Detector发展到现在，我们之前介绍的RCNN/SSP/Fast RCNN/Faster RCNN等，都是属于two-stage detectors，意思就是说**第一步需要生成region proposals，第二步再将整个detection转化为对这些region proposals的classification问题来做**。那所谓的one-stage detection就自然是不需要生成region proposals了，而是直接输出bbox了。Faster RCNN里面作者已经分析了，two-stage detection为啥慢？很大原因就是因为region proposal generation太慢了(例如Selective Search算法)，所以提出了RPN来辅助生成region proposals。

### What is SSD?
SSD最主要的改进就是**使用了一个小的Convolution Filter来预测object category和bbox offset**。那如何处理多尺度问题呢？SSD采取的策略是将这些conv filter应用到多个feature map上，来使得整个模型对Scale Invariant。

![SSD Framework](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-detection/SSD.png)

### Details of SSD
SSD主要部件如下：一个DCNN用来提取feature，产生fixed-size的bbox以及这些bbox中每个类的presence score，然后NMS用来输出最后的检测结果。Feature Extraction部分和普通的分类DCNN没啥太大的区别，作者在后面新添加了新的结构：
1. **Multi-scale feature maps for detection**: 在base feature extraction network之后额外添加新的conv layers(所以得到了multi-scale的feature maps)，来使得模型可以处理multi-scale的detection。
2. **Convolutional predictors for detection**: 每一个新添加的feature layer可以基于```small conv filters```产生fixed-size detection predictions。
3. **Default boxes and aspect ratios**: 对于每个feature map cell，算法给出cell中default box的relative offset，以及class-score(表示在每个box中一个class instance出现的概率)。具体的，对于每个given location的$k$个box，产生4个bbox offset和$c$个class score，这样就对每个```feature map location```上产生了$(c+4)k$个filters，那么对于一个$m\times n$的```feature map```，则产生$(c+4)kmn$个output。这个做法和Faster RCNN中的anchor box有点类似，但是**SSD中将它用到了多个不同resolution的feature map上，因此多个feature map的不同default box shape使得我们可以很高效地给出output box shape**。

![SSD and YOLO](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-detection/SSD_YOLO.png)

#### Training of SSD
前面已经提到了，SSD会在default box周围生成一系列varied location/ratio/scale的boxes，那么到底哪一个box才是和groundtruth真正匹配的呢？作者采用了这样的一个Matching Strategy: 对于每一个groundtruth box，我们从default boxes中选择不同location/ratio/scale的boxes，然后计算它们和任意一个groundtruth box的Jaccard Overlap，并挑选出超过阈值的boxes。那么SSD最终的Loss就可以写成：
$$
L(x,c,l,g)=\frac{1}{N}(L_{conf}(x,c) + \alpha L_{loc}(x,l,g))
$$
其中$N$代表matched default boxes的数量，Localisation Loss和Faster RCNN一样也选择了Smooth L1。$x_{ij}^p=\{0,1\}$为第$i$个default box是否和第$j$个groundtruth box匹配的indicator。
$$
L_{loc}(x,l,g)=\sum_{i\in Pos} \sum_{m\in \{cx,cy,w,h\}}x_{ij}^k smoothL_1(l_i^m-\hat{g}_j^m)
$$

$$
\hat{g}_j^{cx}=(g_j^{cx}-d_i^{cx})/d_i^w
$$

$$
\hat{g}_j^{cy}=(g_j^{cy}-d_i^{cy})/d_i^h
$$

$$
\hat{g}_j^w=log(\frac{g_j^w}{d_i^w})
$$

$$
\hat{g}_j^h=log(\frac{g_j^h}{d_i^h})
$$

Confidence Loss采用Softmax Loss:
$$
L_{conf}(x, c)=-\sum_{i\in Pos}^N x_{ij}^p log(\hat{c}_i^0)-\sum_{\in Neg}log(\hat{c}_i^0)
$$
其中，$\hat{c}_i^p=\frac{exp(c_i^p)}{\sum_p exp(c_i^p)}$

SSD在检测large object时效果很好，但是在检测small object时则效果比较差，这是因为在higher layers，feature map包含的small object信息太少，可通过将input size由$300\times 300$改为$512\times 512$，**Zoom Data Augmentation**(即采用zoom in来生成large objects, zoom out来生成small objects)来进行一定程度的缓解。


## Light-head RCNN
> Paper: [Light-Head R-CNN: In Defense of Two-Stage Object Detector](https://arxiv.org/pdf/1711.07264v2.pdf)

### Introduction
在介绍Light-head RCNN之前，我们先来回顾一下常见的two-stage detector为什么是heavy-head？作者发现two-stage detector之所以慢，就是因为two-stage detector在RoI Warp前/后 会进行非常密集的计算，例如Faster RCNN包含2个fully connected layers做nRoI Recognition，RFCN会生成很大的score maps。所以无论你的backbone network使用了多么精巧的小网络结构，但是总体速度还是提升不上去。所以针对这个问题，作者在本文提出了```light-head``` RCNN。所谓的```light-head```，其实说白了就是```使用thin feature map + cheap RCNN subnet (pooling和单层fully connected layer)```。

大家都知道，two-stage detector，其实是将detection问题转化为一个classification问题来完成的。也就是说，在第一个stage，模型会生成很多region proposal (此为```body```)，然后在第二个stage对这些region proposal进行分类 (此为```head```)。通常，two-stage detector的accuracy要比one-stage detector高的，所以为了accuracy，head往往会设计得非常heavy。Light-head RCNN是这么做的：
> In this paper, we propose a light-head design to build an efficient yet accurate two-stage detector. Specifically, we apply a large-kernel separable convolution to produce "thin" feature maps with small channel number ($\alpha \times p\times p$ is used in our experiments and $\alpha\leq 10$). This design greatly reduces the computation of following RoI-wise subnetwork and makes the detection system memory-friendly. A cheap single fully-connected layer is attached to the pooling layer, which well exploits the feature representation for classification and regression.

### Delve Into Light-Head RCNN
#### RCNN Subnet
> Faster R-CNN adopts a powerful R-CNN which utilizes two large fully connected layers or whole Resnet stage 5 [28, 29] as a second stage classifier, which is beneficial to the detection performance. Therefore Faster R-CNN and its extensions perform leading accuracy in the most challenging benchmarks like COCO. However, the computation could be intensive especially when the number of object proposals is large. To speed up RoI-wise subnet, **R-FCN first produces a set of score maps for each region, whose channel number will be $classes\_num\times p \times p$ ($p$ is the followed pooling size), and then pool along each RoI and average vote the final prediction. Using a computation-free R-CNN subnet, R-FCN gets comparable results by involving more computation on RoI shared score maps generation**.

Faster RCNN虽然在RoI Classification上表现得很好，但是它需要global average pooling来减小第一个fully connected layer的计算量，```而GAP会影响spatial localization```。此外，Faster RCNN对每一个RoI都要feedforward一遍RCNN subnet，所以在当proposal的数量很大时，效率就非常低了。

#### Thin Feature Maps for RoI Warping
在feed region proposal到RCNN subnet之前，用RoI warping来得到fixed shape的feature maps。本文提出的light-head产生了一系列```thin feature maps```，然后再接RoI Pooling层。在实验中，作者发现```RoI warping on thin feature maps```不仅仅提高了精度，而且节省了training和inference的时间。而且，如果直接应用RoI pooling到thin feature maps上，一方面模型可以减少计算量，另一方面可以去掉GAP来保留spatial information。

![Light Head RCNN](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-detection/light_head_rcnn.jpg)

### Experiments
作者在实验中发现，regression loss比classification loss要小很多，所以```将regression loss的权重进行double来balance multi-task training```。


## YOLO v1
> Paper: [You Only Look Once: Unified, Real-Time Object Detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)

YOLO是One-stage Detection领域里一个非常著名的算法，本文对其V1版本做一个简介。
和基于Slide Window/Region Proposal based two-stage detection不同的是，YOLO可以将整张图作为输入（相比之下，two-stage detector需要在第一步先基于selective search等算法生成region proposal，再基于这些region proposal去做classification），**因此YOLO可以获取context information**，而Fast RCNN则很容易将background patch误识为object，原因就在于基于region proposal的detector不能获知larger context information。而YOLO则可以很好地解决该问题。

### What is YOLO?
YOLO将input image先划分为$S\times S$个grid，**若某个object的中心落在了一个grid cell，那么该grid cell就“负责”检测该物体**。每个grid cell预测$B$个bbox以及相对应的confidence score，我们将confidence定义为：
$$
Pr(Object)\star IOU_{pred}^{truth}
$$
若该grid cell没有object，则confidence score自然就为0了，**confidence score为prediction bbox与gt bbox的IOU**。

此外，每个grid cell也预测$C$个conditional class probabilities $Pr(Class_i|Object)$，在测试阶段，我们将conditional class probability和bbox confidence score相乘：
$$
Pr(Class_i|Object)\times Pr(Object)\times IOU_{pred}^{truth}=Pr(Class_i)\times IOU_{pred}^{truth}
$$
这样就得到了对每个box的class-specific confidence score，即同时encode了object在该box中的probability和bbox prediction的精度。

![YOLO V1 Model](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-detection/yolo_v1.jpg)

#### Training of YOLO
因detection需要非常细粒度的feature，所以作者将input resolution由$224\times 224$提升至$448\times 448$来使得网络可以更好地捕捉到细节特征。同时将bbox width/height除以image width/height来进行归一化到$[0, 1]$区间。

> We optimize for sum-squared error in the output of our model. We use sum-squared error because it is easy to optimize **however it does not perfectly align with our goal of maximizing average precision. It weights localization error equally with classification error which may not be ideal. Also, in every image many grid cells do not contain any object. This pushes the confidence scores of those cells towards zero, often overpowering the gradient from cells that do contain objects. This can lead to model instability, causing training to diverge early on**.

针对上述问题，作者使用了两个hyper-param $\lambda_{coord}=5$和$\lambda_{noobj}=0.5$来控制bbox no-object prediction loss和confidence loss。

> Sum-squared error also equally weights errors in large boxes and small boxes. Our error metric should reflect that small deviations in large boxes matter less than in small boxes. To partially address this we predict the square root of the bounding box width and height instead of the width and height directly.

YOLO的Loss如下：
$$
\lambda_{coord}\sum_{i=0}^{S^2} \sum_{j=0}^{B}\mathbb{I} _{ij}^{obj}[(x_i-\hat{x}_i)^2+(y_i-\hat{y}_i)^2]
$$

$$
+\lambda_{coord}\sum_{i=0}^{S^2} \sum_{j=0}^{B}\mathbb{I} _{ij}^{obj}[(\sqrt{w_i}-\sqrt{\hat{w}_i})^2+ (\sqrt{h_i}-\sqrt{\hat{h}_i})^2]
$$

$$
+\sum_{i=0}^{S^2} \sum_{j=0}^{B}\mathbb{I}_{ij}^{obj}(C_i-\hat{C}_i)^2 + \lambda_{noobj}\sum_{i=0}^{S^2} \sum_{j=0}^{B}\mathbb{I}_{ij}^{noobj}(C_i-\hat{C}_i)^2
$$

$$
+\sum_{i=0}^{S^2}\mathbb{I}_{i}^{obj}\sum_{c\in classes} (p_i(c)-\hat{p}_i(c))^2
$$

where $\mathbb{I}_{i}^{obj}$ denotes if object appears in cell $i$ and denotes that the $j$-th bounding box predictor in cell $i$ is responsible for that prediction.

> Note that the loss function only penalizes classification error if an object is present in that grid cell (hence the conditional class probability discussed earlier). It also only penalizes bounding box coordinate error if that predictor is responsible for the ground truth box (i.e. has the highest IOU of any predictor in that grid cell). **However, some large objects or objects near the border of multiple cells can be well localized by multiple cells. Non-maximal suppression can be used to fix these multiple detections**.

#### Limitations of YOLO
> Our model also uses relatively coarse features for predicting bounding boxes since our architecture has multiple downsampling layers from the input image. Finally, while we train on a loss function that approximates detection performance, **our loss function treats errors the same in small bounding boxes versus large bounding boxes. A small error in a large box is generally benign but a small error in a small box has a much greater effect on IOU. Our main source of error is incorrect localizations**.

![Error Analysis](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-detection/yolo_v1_error_analysis.jpg)


## YOLO V2
> [YOLO9000: Better, Faster, Stronger.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf)





## Reference
1. Girshick, Ross, et al. ["Rich feature hierarchies for accurate object detection and semantic segmentation."](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.
2. He, Kaiming, et al. ["Spatial pyramid pooling in deep convolutional networks for visual recognition."](https://arxiv.org/pdf/1406.4729v4.pdf) European conference on computer vision. Springer, Cham, 2014.
3. Girshick, Ross. ["Fast r-cnn."](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf) Proceedings of the IEEE international conference on computer vision. 2015.
4. Ross, Tsung-Yi Lin Priya Goyal, and Girshick Kaiming He Piotr Dollár. ["Focal Loss for Dense Object Detection."](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)
5. Ren, Shaoqing, et al. ["Faster r-cnn: Towards real-time object detection with region proposal networks."](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf) Advances in neural information processing systems. 2015.
6. Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C. Y., & Berg, A. C. (2016, October). [Ssd: Single shot multibox detector](https://arxiv.org/pdf/1512.02325v5.pdf). In European conference on computer vision (pp. 21-37). Springer, Cham.
7. Redmon, Joseph, et al. ["You only look once: Unified, real-time object detection."](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
8. Li Z, Peng C, Yu G, et al. [Light-head r-cnn: In defense of two-stage object detector](https://arxiv.org/pdf/1711.07264v2.pdf)[J]. arXiv preprint arXiv:1711.07264, 2017.
9. Lin, Tsung-Yi, et al. ["Feature Pyramid Networks for Object Detection."](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf) CVPR. Vol. 1. No. 2. 2017.
10. Redmon, Joseph, and Ali Farhadi. ["YOLO9000: Better, Faster, Stronger." ](http://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf)2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2017.