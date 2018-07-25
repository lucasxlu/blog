---
title: "[DL] BackPropogation"
date: 2018-07-25 14:33:25
mathjax: true
tags:
- Machine Learning
- Deep Learning
- Optimization
- Data Science
catagories:
- Algorithm
- Machine Learning
- Deep Learning
- Optimization
---
## Introduction
反向传播是神经网络训练过程中非常重要的步骤。目前许多深度学习框架以（例如Tensorflow）已在定义的computational graph中自行帮开发者完成了反向传播算法的计算。但是作为深度学习领域的研究人员，还是应该了解该算法的本质。本文就对该算法进行深入讲解（素材来自Stanford CS231n Spring,2017）：  
一个简单的computational graph $f(x, y, z)=(x + y)z$ (e.g. $x = -2$, $y = 5$, $z = -4$)是这样的：  


令q=x+y，则∂q/∂x=1, ∂q/∂y=1 可得f=qz，则∂f/∂q=z=-4,    ∂f/∂z=q=3


