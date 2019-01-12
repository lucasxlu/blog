---
title: "[CV] Counting"
date: 2019-01-12 22:44:44
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

> [@LucasX](https://www.zhihu.com/people/xulu-0620/activities)注：本文长期更新。



## Reference
1. Shen, Zan, et al. ["Crowd Counting via Adversarial Cross-Scale Consistency Pursuit."](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Crowd_Counting_via_CVPR_2018_paper.pdf) Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
2. Liu, Xialei, Joost van de Weijer, and Andrew D. Bagdanov. ["Leveraging Unlabeled Data for Crowd Counting by Learning to Rank."](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Leveraging_Unlabeled_Data_CVPR_2018_paper.pdf) Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
3. Marsden, Mark, et al. ["People, Penguins and Petri Dishes: Adapting Object Counting Models To New Visual Domains And Object Types Without Forgetting."](http://openaccess.thecvf.com/content_cvpr_2018/papers/Marsden_People_Penguins_and_CVPR_2018_paper.pdf) Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
4. Liu, Jiang, et al. ["Decidenet: Counting varying density crowds through attention guided detection and density estimation."](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_DecideNet_Counting_Varying_CVPR_2018_paper.pdf) Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
5. Hinami, Ryota, Tao Mei, and Shin'ichi Satoh. ["Joint detection and recounting of abnormal events by learning deep generic knowledge."](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hinami_Joint_Detection_and_ICCV_2017_paper.pdf) Proceedings of the IEEE International Conference on Computer Vision. 2017.
6. Hsieh, Meng-Ru, Yen-Liang Lin, and Winston H. Hsu. ["Drone-based object counting by spatially regularized regional proposal network."](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hsieh_Drone-Based_Object_Counting_ICCV_2017_paper.pdf) The IEEE International Conference on Computer Vision (ICCV). Vol. 1. 2017.
7. Zhang, Yingying, et al. ["Single-image crowd counting via multi-column convolutional neural network."](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
8. Sam, Deepak Babu, Shiv Surya, and R. Venkatesh Babu. ["Switching convolutional neural network for crowd counting."](http://openaccess.thecvf.com/content_cvpr_2017/papers/Sam_Switching_Convolutional_Neural_CVPR_2017_paper.pdf) Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. Vol. 1. No. 3. 2017.
9. Chattopadhyay, Prithvijit, et al. ["Counting everyday objects in everyday scenes."](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chattopadhyay_Counting_Everyday_Objects_CVPR_2017_paper.pdf) Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.
10. Ranjan, Viresh, Hieu Le, and Minh Hoai. ["Iterative crowd counting."](http://openaccess.thecvf.com/content_ECCV_2018/papers/Viresh_Ranjan_Iterative_Crowd_Counting_ECCV_2018_paper.pdf) Proceedings of the European Conference on Computer Vision (ECCV). 2018.
11. Cao, Xinkun, et al. ["Scale aggregation network for accurate and efficient crowd counting."](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xinkun_Cao_Scale_Aggregation_Network_ECCV_2018_paper.pdf) Proceedings of the European Conference on Computer Vision (ECCV). 2018.