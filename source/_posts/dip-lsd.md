---
title: "[DIP] LSD"
date: 2020-08-16 16:33:37
mathjax: true
tags:
- Digital Image Processing
- Computer Vision
- Line Segment Detection
catagories:
- Digital Image Processing
- Computer Vision
- Line Segment Detection
---
## Line Segmentation Detection
LSD算法是一个直线提取算法，在opencv 中也有封装，它的主要思想是通过求导得到灰度的梯度，因为灰度梯度的方向的垂直方向就是线条的方向，将有相同方向的向量用矩形圈起来，再将举行精细化，最后就可以得到的一条线段了。

首先，我们看下图，这里的图片首先根据梯度的垂线构造了一个level-line field，它把和梯度的垂直方向的线叫做level-line，所以这个field就是由这样的线组成的场。有了这个场，因为有直线的区域我们总能够找到，很多方向相同的level-line，这样我们就能得到右边这些颜色区域了，也就是所谓的line support region。
![LSD-1](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dip-lsd/lsd1.png)

如果把这个区域用举行圈起来的话就得到了如下图所示的矩形，注意这个举行我们是区分方向的。
![LSD-2](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dip-lsd/lsd2.png)

至于，是根据什么规则生成的矩形，就像之前说的，在直线的区域，自然地level-line(梯度方向的垂线）自然就会比较接近，因此，我们就给定一个阈值/𝑡𝑎𝑢，如果这些向量的夹角小于/𝑡𝑎𝑢且在这个区域内满足条件的向量足够多，我们就认为这个区域可能是直线，比如下图中，就只有8个符合条件的level-line，感觉是不足够多的，因此它可能不被看作是一个直线。具体地，这里使用了区域生长算法，即从某个点开始，找其领域内符合条件的点。
![LSD-3](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dip-lsd/lsd3.png)

LSD算法流程如下：
![LSD-Algorithm](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dip-lsd/lsd_algo.png)

具体操作步骤为：
1. 缩放至80% (Guassian Subsampling)
2. 计算horizontal & vertical方向的gradient
3. gradient pseudo-ordering:
   > LSD is a greedy algorithm and the order in which pixels are processed has an impact on the result. Pixels with high gradient magnitude correspond to the more contrasted edges. In an edge, the central pixels usually have the highest gradient magnitude. So it makes sense to start looking for line segments at pixels with the highest gradient magnitude.
   >
   > Sorting algorithms usually require O(n log n) operations to sort n values. However,a simple pixel pseudo-ordering is possible in linear-time. To this aim, 1024 bins are created corresponding to equal gradient magnitude intervals between zero and the largest observed value on the image. Pixels are classied into the bins according to their gradient magnitude. LSD uses rst seed pixels from the bin of the largest gradient magnitudes; then it takes seed pixels from the second bin, and so on until exhaustion of all bins. 1024 bins are enough to sort almost strictly the gradient values when the gray level values are quantized in the integer range $[0,255]$.
4. gradient threshold:
   > Pixels with small gradient magnitude correspond to at zones or slow gradients. Also, they naturally present a higher error in the gradient computation due to the quantization of their values. In LSD the pixels with gradient magnitude smaller than  are therefore rejected and not used in the construction of line-support regions or rectangles.
5. region growing:
   ![LSD-Region-Growing](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/dip-lsd/lsd_region_grow.png)
6. rectangular approximation
7. NFA computation:
   > A key concept in the validation of a rectangle is that of $p$-aligned points, namely the pixels in the rectangle whose level-line angle is equal to the rectangle's main orientation, up to a tolerance $p\pi$.

   $$
   NFA = (NM)^{5/2}\gamma \sum_{j=k}^n \begin{pmatrix} n \\ j \end{pmatrix}p^j (1-p)^{n-j}
   $$

   **The rectangles with $NFA(r)\leq \epsilon$ are validated as detections.**
8. Aligned Points Density
    - reduce angle tolerance
    - reduce region radius
9. rectangle improvement



## References
1. Von Gioi R G, Jakubowicz J, Morel J M, et al. [LSD: a line segment detector](https://www.ipol.im/pub/art/2012/gjmr-lsd/article.pdf)[J]. Image Processing On Line, 2012, 2: 35-55.
2. Von Gioi R G, Jakubowicz J, Morel J M, et al. [LSD: A fast line segment detector with a false detection control](https://ieeexplore.ieee.org/abstract/document/4731268/)[J]. IEEE transactions on pattern analysis and machine intelligence, 2008, 32(4): 722-732.