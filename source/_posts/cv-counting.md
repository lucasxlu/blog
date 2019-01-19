---
title: "[CV] Counting"
date: 2019-01-19 14:38:44
mathjax: true
tags:
- Computer Vision
- Machine Learning
- Deep Learning
catagories:
- Computer Vision
- Machine Learning
- Deep Learning
---
## Introduction
Counting是近年来CV领域一个受到关注越来越多的方向，它主要的应用场景就是密集场景下的人流估计、车辆估计等。Counting大体上可以分为两种方案，一种是基于detection的方式：即数bbox；另一种是直接回归density map的方式：即将counting问题转化为一个regression问题。基于detection的方法在目标非常密集的场景下就不适合了，所以在这种场景下density map regression还是目前的mainstream。

Counting的Metric通常为MAE和MSE，MAE评判counting heads的accuracy，MSE评判robustness。


> [@LucasX](https://www.zhihu.com/people/xulu-0620/activities)注：本文长期更新。


## Multi-Column CNN
> Paper: [Single-image crowd counting via multi-column convolutional neural network](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)

MCNN (Multi-Column CNN)主要idea来自于2012年上的一篇CVPR Paper [Multi-column Deep Neural Networks for Image Classification](https://arxiv.org/pdf/1202.2745.pdf)，什么叫```multi-column```呢？意思就是给定一张图，分3个并行的branch，每个branch采用不同的conv size/pooling stride等，最后用$1\times 1$ conv对这3个不同branch输出的feature map做linear weighted combination。注：Counting属于content geometry-sensitive task，一般不直接warp到fixed size。

MCNN有以下好处：
* 通过3个receptive fields不同(small/medium/large)column，每个column可以更好地适应由不同图像分辨率带来的不同尺度问题。
* 在MCNN中，作者将最后一层的FC替换为了$1\times 1$的conv layer，因此**input image可以为任意尺寸，而不需要强行warp导致变形**。网络的输出就是density estimate。

### Delve into MCNN
#### Density map based crowd counting
通过DCNN来进行crowd counting主要有以下两种思路：第一种是直接回归(输入image，输出counting number)；第二种是输出crowd的density map(即每一平米有多少head count)，然后通过integration的方式进行head count计算。MCNN选用了第二种方式(即基于density map)，这样做有如下优点：
* Density map保留了更多的信息；与直接regress head count相比，density map保留了input image中crowd的spatial distribution，而这种spatial distribution information在很多任务中都是非常有用的。例如某个small region的density比其他regions要高很多，那么就可以说明这个region出现了something abnormal。
* 通过CNN学习density map，learned filters可以更好地适应于不同size的human head(因为实际场景中因拍摄角度、距离等因素，会造成人头尺寸在图片中不同的大小)，因此这些learned filters会更加semantic meaningful，进而提高整个模型的performance。

若pixel $x_i$有head，我们将其表示为一个函数$\delta(x-x_i)$，因此有$N$个labeled head的图片可以表示为：
$$
H(x)=\sum_{i=1}^N \delta(x-x_i)
$$

为了将其转换为一个连续密度函数，我们使用Gaussian Kernel$G_{\sigma}$来对$H(x)$进行卷积，所以density就成了$F(x)=H(x)\star G_{\sigma}(x)$。然而，这样的density function需要假定**在图像空间中$x_i$都是互相独立的**。这显然是不太符合实际的，实际上，每个$x_i$都是3D scene中因perspective distortion带来的crowd density的一个样本，并且这些pixels和scene中对应不同size的样本$x_i$都是有关联的。

因此，为了准确地预估crowd density $F$，我们需要考虑因ground plane和image plane的之间的homography带来的distortion问题。但是对于图片数据本身而言，我们并无法知道geometry scene是怎样的。然而，若我们假设每个head周围的crowd都是evenly distributed，那么这些head和其k近邻的平均距离就可以得到关于geometry distortion合理的预估。

因此，我们需要根据图片中每个person head的size来确定spread parameter $\sigma$。但现实中很难精确知道head size的大小，以及head size和density map的关系。作者发现，**head size is relative to the distance between the centers of two neighboring persons in crowd scene**。因此对于这些crowded scene的density map，**可基于每个person head与其k近邻的平均距离**来adaptively determine spread parameter。

对于图片中的每个head $x_i$，我们定义其k近邻为$\{d_1^i,d_2^i,\cdots,d_m^i\}$，平均距离为$\bar{d^i}=\frac{1}{m}\sum_{j=1}^m d_j^i$。因此，和$x_i$相关像素所对应在scene ground中的区域，与$\bar{d^i}$大致成半径比例。因此，为了估计pixel $x_i$周围的crowd density，我们用以与$\bar{d^i}$成比例的$\sigma_i$为variance的Gaussian Kernel去对$\delta(x-x_i)$进行卷积。Density $F$为：
$$
F(x)=\sum_{i=1}^N \delta(x-x_i)\star G_{\sigma_i}(x), \sigma_i=\beta \bar{d^i}
$$
> for some parameter $\beta$. In other words, we convolve the labels $H$ with density kernels adaptive to the local geometry around each data point, referred to as geometry-adaptive kernels.

#### Multi column CNN for density map estimation
因perspective distortion，图片通常会包含各种不同size的head，因此不同receptive fields的conv filter能够capture到不同scale的crowd density。对应large receptive fields的conv filter对于estimate large head size的crowd更有用。

![MCNN](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-counting/MCNN.jpg)

那么如何将feature map映射到density map呢？
作者是这样做的：将CNN所有的output feature maps堆叠起来，并使用$1\times 1$进行conv(实际上就是linear weighted combination；这个方法和Point-wise Convolution比较类似)。然后用Euclidean distance来measure estimated density map和groundtruth之间的difference。
$$
L(\Theta)=\frac{1}{2N}\sum_{i=1}^N\|F(X_i;\Theta)-F_i\|_2^2
$$

有这样一些data tricks值得mention一下：
* 传统CNN往往会normalize input来得到fixed size image，但是MCNN支持arbitrary size的input。**因为resize会带来density map中的distortion问题，而这对于crowd density estimate是非常不利的**。
* 和传统[Multi-column](https://arxiv.org/pdf/1202.2745.pdf)粗暴地将feature map进行average相比，MCNN通过$1\times 1$ conv对其进行linear weighted combination。
* 在实验中，作者先单独pre-train 每个column，然后再将3个column进行fine-tune。


## Leveraging Unlabeled Data for Crowd Counting by Learning to Rank
> Paper: [Leveraging Unlabeled Data for Crowd Counting by Learning to Rank](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Leveraging_Unlabeled_Data_CVPR_2018_paper.pdf)

这篇文章将counting问题转化为了一个ranking问题，主要idea就是说**在一张图中，crop下来的部分中包含的crowd number是肯定要不多于原图的crowd number**。然后利用这个作为supervision signal，将其转化成为ranking问题。这样做就可以用到大量未标注crowd density的图片来做self supervised learning。

![Using ranked sub-images for self-supervised training](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-counting/ranking_counting.jpg)

Counting近些年来得到了很大的关注，但是给这些人群密集的图片打label是非常麻烦的。最近self-supervised learning得到了越来越多的关注，它可以从auxiliary task(different but related to original supervised task)中进行学习。

本文提出的Learning to rank主要基于这样一个idea：尽管我们没办法获知一张crowd image中精确的crowd number，但是可以确定的是，**crop samples from a crowd image contain the same or fewer persons than the original**。这样我们就可以产生一系列sub-images的ranking，去训练网络预估一张图是否比另一张图包含更多的persons。

### Generating ranked image sets for counting
* Input: A crowd scene image, number of patches $k$ and scale factor $s$.
* A list of patches ordered according to the number of persons in the patch.

Step 1: Choose an anchor point randomly from the anchor region. The anchor region is defined to be $1/r$ the size of the original image, centered at the original image center, and with the same aspect ratio as the original image.

Step 2: Find the largest square patch centered at the anchor point and contained within the image boundaries.

Step 3: Crop $k−1$ additional square patches, reducing size iteratively by a scale factor s. Keep all patches centered at anchor point.

Step 4: Resize all $k$ patches to input size of network.

### Learning from ranked image sets
#### Crowd density estimation network
作者使用VGG16作为backbone network来回归crowd density map，但是去掉了两个fully connected layers和pool5来保留住更多的spatial resolution，并且增加了一个$3\times 3\times 512$和zero padding来保证same size。这里直接使用Euclidean distance来作为loss：
$$
L_{c}=\frac{1}{M}\sum_{i=1}^M (y_i-\hat{y}_i)^2
$$
$y_i$是groundtruth person density map，$\hat{y}_i$是prediction。

Crowd counting的groundtruth通常包含一系列坐标点，代表**head center of a person**。为了将其转换为crowd density map，我们用标准差为15 pixels的Gaussian并将scene中所有的persons进行求和来得到$y_i$。

网络结构如下：
![The multi-task framework combining both counting and ranking information](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/cv-counting/ranking_counting_framework.jpg)

此外，为了进一步提高性能，作者采用了multi-scale sampling策略，即使用56——448 pixels之间varying size的square patches作为输入，作者发现这种multi-scale input可以显著提高模型性能。

#### Crowd ranking network
对于ranking network，我们将Euclidean loss替换为average pooling layer + ranking loss。首先，average pooling layer将density map转换为每个spatial unit $\hat{c}(I_i)$中person number的estimate。其中，
$$
\hat{c}(I_i)=\frac{1}{M}\sum_{j}\hat{y}_i (x_j)
$$
> $x_j$ are the spatial coordinates of the density map, and $M = 14\times 14$ is the number of spatial units in the density map. The ranking which is on the total number of persons in the patch $\hat{C}_i$ also directly holds for its normalized version $\hat{c}_i$, since $\hat{C}(I_i)=M\times \hat{c}(I_i)$.

这里使用了pairwise ranking hinge loss：
$$
L_{r}=max(0, \hat{c}(I_2) - \hat{c}(I_1) + \epsilon)
$$
$\epsilon$是margin，但是在本文中设为0，我们假定 $\hat{c}(I_1)$ 的rank比 $\hat{c}(I_2)$ 高。

Ranking loss的梯度计算如下：
$$
\triangledown_{\theta}L_r=\begin{cases}
    0 & \hat{c}(I_2) - \hat{c}(I_1) + \epsilon\leq 0\\
    \triangledown_{\theta}\hat{c}(I_2) - \triangledown_{\theta}\hat{c}(I_1) & otherwise
\end{cases}
$$

> When network outputs the correct ranking there is no backpropagated gradient. However, when the network estimates are not in accordance with the correct ranking the backpropagated gradient causes the network to increase its estimate for the patch with lower score and to decrease its estimate for the one with higher score (note that in backpropagation the gradient is subtracted).

在实现中，作者将图片作为one batch(for regression & ranking)更为有效，因此ranking loss可以被计算为：
$$
L_r=\sum_{i=1}^M \sum_{j\in S(i)}max(0, \hat{c}(I_j) - \hat{c}(I_i) + \epsilon)
$$
> where $S(i)$ is the set of patches containing fewer people than patch $i$. Note that this relation is only defined for patches which are contained by patch $i$. In practice we sample minibatches of 25 images which contain 5 sets of 5 images which can be compared among them resulting in a total of $5\times (4 + 3 + 2 + 1) = 50$ pairs in one minibatch.

#### Combining counting and ranking data
Joint loss的叠加有以下3中方案：
* **Ranking plus fine-tuning**: In this approach the network is first trained on the large dataset of ranking data, and is next fine-tuned on the smaller dataset for which density maps are available.
* **Alternating-task training**: While ranking plus finetuning works well when the two tasks are closely related, it might perform bad for crowd counting because no supervision is performed to indicate what the network is actually supposed to count. Therefore, we propose to alternate between the tasks of counting and ranking. In practice we perform train for 300 minibatches on a single task before switching to the other, then repeat.
* **Multi-task training**: In the third approach, we add the self-supervised task as a proxy to the supervised counting task and train both simultaneously.
$$
L=L_c + \lambda L_r
$$





## Reference
1. Zhang, Yingying, et al. ["Single-image crowd counting via multi-column convolutional neural network."](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
2. Shen, Zan, et al. ["Crowd Counting via Adversarial Cross-Scale Consistency Pursuit."](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Crowd_Counting_via_CVPR_2018_paper.pdf) Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
3. Liu, Xialei, Joost van de Weijer, and Andrew D. Bagdanov. ["Leveraging Unlabeled Data for Crowd Counting by Learning to Rank."](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Leveraging_Unlabeled_Data_CVPR_2018_paper.pdf) Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
4. Marsden, Mark, et al. ["People, Penguins and Petri Dishes: Adapting Object Counting Models To New Visual Domains And Object Types Without Forgetting."](http://openaccess.thecvf.com/content_cvpr_2018/papers/Marsden_People_Penguins_and_CVPR_2018_paper.pdf) Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
5. Liu, Jiang, et al. ["Decidenet: Counting varying density crowds through attention guided detection and density estimation."](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_DecideNet_Counting_Varying_CVPR_2018_paper.pdf) Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
6. Hinami, Ryota, Tao Mei, and Shin'ichi Satoh. ["Joint detection and recounting of abnormal events by learning deep generic knowledge."](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hinami_Joint_Detection_and_ICCV_2017_paper.pdf) Proceedings of the IEEE International Conference on Computer Vision. 2017.
7. Hsieh, Meng-Ru, Yen-Liang Lin, and Winston H. Hsu. ["Drone-based object counting by spatially regularized regional proposal network."](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hsieh_Drone-Based_Object_Counting_ICCV_2017_paper.pdf) The IEEE International Conference on Computer Vision (ICCV). Vol. 1. 2017.
8. Sam, Deepak Babu, Shiv Surya, and R. Venkatesh Babu. ["Switching convolutional neural network for crowd counting."](http://openaccess.thecvf.com/content_cvpr_2017/papers/Sam_Switching_Convolutional_Neural_CVPR_2017_paper.pdf) Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. Vol. 1. No. 3. 2017.
9.  Chattopadhyay, Prithvijit, et al. ["Counting everyday objects in everyday scenes."](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chattopadhyay_Counting_Everyday_Objects_CVPR_2017_paper.pdf) Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.
10. Ranjan, Viresh, Hieu Le, and Minh Hoai. ["Iterative crowd counting."](http://openaccess.thecvf.com/content_ECCV_2018/papers/Viresh_Ranjan_Iterative_Crowd_Counting_ECCV_2018_paper.pdf) Proceedings of the European Conference on Computer Vision (ECCV). 2018.
11. Cao, Xinkun, et al. ["Scale aggregation network for accurate and efficient crowd counting."](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xinkun_Cao_Scale_Aggregation_Network_ECCV_2018_paper.pdf) Proceedings of the European Conference on Computer Vision (ECCV). 2018.