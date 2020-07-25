---
title: "[Note] AI Industry"
date: 2020-07-25 11:55:16
mathjax: true
tags:
- Machine Learning
- Deep Learning
- Business
- Artificial Intelligence
catagories:
- Machine Learning
- Deep Learning
- Business
- Artificial Intelligence
---
## Introduction
最近刷知乎，有看到这样一个问题[“如何看待2021年秋招算法岗灰飞烟灭？”](https://www.zhihu.com/question/406974583/answer/1343041027)，虽然标题夸张，但确实反映了当前算法岗位内卷严重的真实情况。前面写了很多篇paper的阅读笔记，这篇就站在产业界的角度，来谈一谈AI在实际产业界的落地、商业模式以及对人才的要求吧。

## What's the Gap Between Academia and Industry?
### Emphasis
在学术界，追求的普遍都是模型的**创新**与方法在某个数据集上的**精度**，因此常见的衡量标准就是Paper产出量以及Kaggle等比赛排名是否靠前。但是在工业界，却并非如此，工业界的第一诉求永远是**能否以最低的成本来解决业务痛点**，**你的模型/策略能带来多大的商业价值**？这个时候，模型的novelty其实并没有那么重要，选择业内成熟的方案，能够低成本稳定运行即可。以CV为例，现如今跑在很多公司服务器上的模型依然是ResNet/Faster RCNN/SphereFace等几年前的“老模型”。这些成熟模型虽然在很多时候其实是已经完全满足了业务需求（因为在工业界，**高质量数据带来的提升远远比算法重要**），但并不是说AI算法工程师就应该停留在舒适区止步不前了。尽管深度学习在近些年遇到了一些瓶颈，但每年依然有一些比较亮眼的工作出现。举个栗子，你现在做图像分类，A同学上来二话不说就是ResNet硬怼，你通过follow最新的research成果，用上了EfficientNet/ECAMobileNet/MobileNetV3等轻量级模型，在准召相近的情况下，你的线上serving只需要两块GPU就可以抗住对应的QPS，而A同学则需要4块GPS，这个时候你的优势是不是就体现出来了？另外就是，在当前AI行业如此内卷的大环境下，你如果能follow最新的paper，并及时赋能实际业务，也表明了你是一个热爱学习的人，面试官往往会觉得眼前一亮（有点“面向简历工作”的意思，哈哈）。**不管什么时候，作为算法工程师，有高质量的paper及Top比赛排名，任何时候都是简历上非常加分的亮点**。


### Deeply Understand Your Business
企业的目的是赚钱，在企业上班的工程师的KPI是产生业务价值，因此绝大多数时候并不会去刻意追求模型的novelty。深入理解业务，了解该业务的上下游，你的模型对上下游的影响，你在整个业务线中处于一个什么样的角色，梳理清晰业务，并从中挖掘出业务上的痛点，并根据你的经验和专业技能提出**低成本的解决方案**。能到达这一步，你就基本达到专家工程师的高度了。很多同学有个错觉，就是觉得算法专家一定要是顶会paper等身才行，而根据我的观察，在企业业务部门，能解决业务痛点更重要。培养业务视角，平时可以多关注行业内大厂前辈们的分享，以及行业研究报告。


## Summary
最后，通过近2年的research经历与2年的工业界经历，我总结了一下视觉算法工程师应该要掌握的知识技能思维导图。感兴趣的同学可以参考，也希望大佬们提意见~

![Computer Vision Algorithm Roadmap](https://raw.githubusercontent.com/lucasxlu/blog/master/source/_posts/ai-industry/Computer_Vision_Algorithm_Roadmap.png)


## Business Model of AI
AI企业主要分为**to B**和**to C**，
**to B**型AI公司主要有以下几种商业模型：
* 为其他企业提供服务（例如SDK、API等），比较知名的有商汤、旷视
* 虽然产品都是自己的，但客户主要是政府、物业等，比较知名的是海康、大华这种智能安防公司
* AI只是辅助功能，以极低的价格（甚至免费）卖调用次数，实际上是卖自己的云服务，例如阿里云、百度云等
* 更有甚者通过开放接口采集数据来迭代自己的模型，形成数据闭环，然后通过迭代的更准确的模型去竞标2B/2G的单子，就不举例了

**to C**型AI公司的服务对象是广大的用户，它们其实并不像前者那样强调自己是AI公司，而是通过AI技术赋能自己的业务，带给用户更好的体验。常见的AI落地场景有：
* 电商：例如淘宝/京东/拼多多的拍照购
* 短视频：例如抖音的视频特效
* 零售：例如AmazonGo的商品识别、根据人流量选址
* ...




## References
1. Sculley, David, et al. ["Hidden technical debt in machine learning systems."](http://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf) Advances in neural information processing systems. 2015.
2. [2021届秋招算法岗真的要灰飞烟灭了吗？](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247494737&idx=1&sn=4ef70cdc11367d93721ad0c9687edf07&chksm=970fc487a0784d91dcd606859c4304dfbb4b1fba223d3d419560e5e0761e4d92a26d407495f9&mpshare=1&scene=24&srcid=0724mweUr2iz5ibmJF6l5Fhq&sharer_sharetime=1595600155408&sharer_shareid=d48f2a1eabae06a1f257160da72857e8&key=ebb412db45555e1d4f384d204d3fde14a2505c7d0d7c375d3f24d88d975b26895cd39983bd24ea1a59ea57d5134381be2ee283a485a8aa96d3486b48772a549d45e26b74804e5de4bd74f1387e6b062b&ascene=14&uin=Mzk3MDE0Nzk1&devicetype=Windows+10+x64&version=62090070&lang=zh_CN&exportkey=ATaJ8%2FY7EqNukjBKQ8hKf8o%3D&pass_ticket=FRSQxc4uErXQddW8g5y7qUbLuFiYNWcnK8QGv14FhMYTbFBI4IkHiIoJlbZJpI1C)