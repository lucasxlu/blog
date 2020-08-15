---
title: "[ML] Distributed Machine Learning"
date: 2020-08-15 10:20:09
mathjax: true
tags:
- Machine Learning
- Deep Learning
- Data Science
- Distributed System
catagories:
- Algorithm
- Machine Learning
- Deep Learning
- Distributed System
---
# Distributed Machine Learning
## 分布式机器学习框架
### 分布式机器学习基本流程
使用分布式机器学习（Distributed Machine Leaning, DML）的情形主要有：
* 计算量太大：可采取基于共享内存（或虚拟内存）的多线程或多机并行计算，即所有workload共享一块公共内存，并且数据和模型全部存储于这块共享内存中，我们不需要对模型和数据进行划分。这时，每个workload都有权访问数据，可以并行执行优化算法，此为``Computation Parallel``
* 训练数据太多：可将数据进行划分，并将数据分配到多个workload上进行训练；每个workload会根据局部数据训练出一个子模型，并且会按照一定的规律和其他workload进行通信，以保证可有效整合各个子模型得到最终全局机器学习模型，即``Data Parallel``
    - 对训练样本进行划分：
      - 随机采样：有放回地进行采样，然后为每个workload的容量分配相应数量的训练样本；这种做法可以保证每台workload上的局部数据与原始训练集是i.i.d，因此训练效果上有理论保证。弊端在于全局采样代价比较高，且某些低频样本很难被选出来，导致某些辛苦标注的样本没有被利用起来。
      - 置乱切分：将训练集进行乱序，然后按照工作节点的个数将打乱后的数据顺序划分成相应的小份，随后将小份数据分配到各个workload上，每个workload在训练过程中只利用分配给自己的局部数据，并且会定期将局部数据再打乱一次。到一定阶段，还可能需要再对全局数据进行shuffle与reassign，这样可使得各个workload上的训练样本更加独立并具有更一致的分布。
  
      **置乱切分相比于随机采样有如下好处：**
        - 置乱切分复杂度比较低，虽然一次置乱切分操作无法实现对数据的完全随机打乱，但可以证明调用若干次后，其置乱效果已可以很好地逼近完全随机。对长度为$n$的序列进行一次置乱操作复杂度为$O(logn)$，而$n$次有放回抽样复杂度为$O(n)$。
        - 置乱切分数据信息量比较大，置乱切分相当于无放回抽样，每个样本都会出现在某个workload上，每个workload的本地数据没有重复，因此用于训练的数据信息量会更大。
    - 对特征维度进行划分：将$d$维向量切分成$K$份，再分配到不同workload去计算，并保证每个workload直接的通信。主要是``Decision Tree``和``Linear Model``。
* 模型规模太大：对模型进行划分，分配到不同的workload进行训练，即``Model Parallel``
    - 横向的逐层划分：例如第$m$层分给第$k$个workload
    - 纵向的跨层划分：将每一层的节点分配给不同的workload
    - 模型随机并行：选取某个骨架网络（因NN参数多，存在冗余），将骨架网络存储于workload，各workload互相通信时，会随机传输一些属于非骨架网络的参数，来起到探索网络全局拓扑结构的作用

DML主要包括``数据与模型划分模块``、``单机优化模块``、``通信模块``以及``数据与模型聚合模块``。

### 通信模块
#### 通信拓扑结构
* 基于迭代式MapReduce/AllReduce的通信拓扑结构：``MapReduce``：``Map``完成数据分发和并行处理，``Reduce``完成数据的全局同步和聚合。MapReduce缺点：完全依赖IO的数据交互使得效率太低；计算过程的中间态不能得到维持，使得反复迭代的机器学习过程无法高效衔接。``迭代式MapReduce``基于内存实现，效率更高，且引入了persistent store来保存中间态。迭代式MapReduce代表系统是``Spark MLlib``、``Cloudera``。
* 基于Parameter Server的通信拓扑结构：PS架构把工作节点和模型存储节点在逻辑上区分开，因此可以更加灵活地支持各种不同通信模式。各个工作节点负责处理本地数据，通过PS客户端API与Parameter Server通信，从而从PS获取最新参数，或者将本地的模型更新push到PS。PS将各个工作节点之间的交互完全隔离开，取而代之的是PS与各节点交互，因此各个工作节点不一定要时刻保持同步，可获得更高的加速比；可使用多个PS来共同维护较大的模型。基于PS的代表系统是``Google DistBelief``。
* 基于Data Flow的通信拓扑结构：计算被描述为一个DAG，graph中的每条edge代表数据流动，当两个节点位于两台不同机器上时，它们之间便会通信。基于Data Flow的通信拓扑结构的代表系统是``Google TensorFlow``。

#### 通信方式
* 同步
* 异步
  * 有锁：各workload可异步本地学习，但把局部信息写入全局模型时，会通过加锁来保证数据写入的完整性
  * 无锁：局部信息写入全局模型时，不保证数据完整性，以换取更好的吞吐量

#### 通信频率
如何降低通信频率：
- 模型压缩
- 模型量化


## 通信机制
### 通信拓扑结构
#### 基于Iterative MapReduce/AllReduce的通信拓扑
- 基于[MapReduce](https://static.googleusercontent.com/media/research.google.com/zh-CN//archive/mapreduce-osdi04.pdf)，``Map``定义数据分发及并行处理，``Reduce``定义全局参数聚合。
- 基于MPI，主要使用``AllReduce``接口来同步任何想要同步的信息。

#### 基于Parameter Server的通信拓扑
当数据量和模型越来越大时，就会出现问题：并行的workload越来越多，且计算性能不均衡时，采用Iterative MapReduce或AllReduce的Distributed System训练速度受制于系统中最慢的workload；若某个workload down掉，则整个系统无法继续运行。

在``Parameter Server``框架中，系统中所有workload被逻辑上分为工作节点（work）和服务器节点（server）。各个work主要负责处理本地训练任务，并通过客户端接口与PS通信，从PS pull最新参数或者将本地更新push到PS上。PS可以由单个服务器担任，也可由多个服务器担任。work和server之间相互通信，各work内部无需通信。

#### 基于Data Flow的通信拓扑
即``Computational Graph``，每个节点进行计算或数据处理，每条边代表数据的流动，当两个节点位于两台不同的机器时，它们之间便会进行通信。每个节点有两个通信通道：``控制消息流``和``计算数据流``。``计算数据流``主要负责接收模型训练时所需要的数据、模型参数等，再经过工作节点内部的计算单元，产生输出数据，按需提供给下游节点。``控制消息流``决定了工作节点应该接收什么数据，接收的数据是否完整，自己要做的计算是否完成，是否可以让下游节点继续计算等。

### 通信的步调
* 同步通信：所有workload以同样的步调进行训练，能保证分布式算法与单机算法的等价性；缺点在于会造成资源闲置浪费。在``Bulk Synchronous Parallel``中，引入了``Barrier Synchronization``，让所有节点在此位置被强制停下，直到所有workload都完成了同步屏障之前的操作，然后系统进行下一步计算，来实现同步计算的效果。
* 异步通信：各workload可按照自己的步调训练，无需彼此等待，从而最大化计算资源的利用率；缺点是会造成各workload的模型不一致。

同步和异步的的平衡处理：``Stale Synchronous Parallel``核心思想是控制最快和最慢节点之间相差的迭代次数不超过预设的阈值。若计算集群中某个workload计算领先太多，则触发等待机制，使得领先的workload的最新参数被挂起，直到慢节点到达当前计算位置后才会将领先的节点解冻。

### 通信的频率
通信频率主要包括``时间频率（即通信频次间隔）``和``空间频率（即通信内容大小）``。相应地，优化通信频率可以从``时域滤波``和``空域滤波``两方面进行。

#### 时域滤波
从通信的过程出发，控制通信的时机，减少通信次数，从而减少通信代价。主要有以下几中方法：

- 增加通信间隔：将通信频率从原来本地模型每次更新后都通信一次，变成本地模型多次更新后才通信一次。
- 非对称的推送和获取
- 计算和传输pipeline：训练线程完成计算部分，传输线程完成网络通信部分。系统中有两个模型缓存buffer1和buffer2，训练过程基于buffer1中的参数产生模型更新，在训练的同时，通信线程先将上轮训练线程产生的更新发送出去，然后获取一份当下最新的全局模型保存在buffer2中。当计算和传输线程都完成一轮操作后，交换两个buffer中的内容：buffer2用于参与训练线程，buffer1用于通信线程发送给Parameter Server。因**模型的训练与网络通信在时间轴上是重叠的，从而减少了总体的时间开销**。

#### 空域滤波
- 模型过滤：若某次迭代中某些参数没有明显变化，便可将其过滤，从而减少通信量。
- 低秩分解：SVD将$M\times N$的大矩阵分解为$M\times k$, $k\times k$以及$k\times N$3个小矩阵，来降低通信量。
- 模型量化：FP32-->FP16/int8/binary weights
- 模型压缩：weight pruning
- Knowledge Distillation
- Design more compact models: ``Neural Architecture Search``, ``MobileNet/ShuffleNet/MixNet/Global Average Pooling/SqueezeNet``


## 数据与模型聚合
### 基于模型加和的聚合方法
#### 基于全部模型加和的方法
#### 基于部分模型加和的方法
  * 带备份节点的同步随机梯度下降法：在MapReduce系统中，当我们发现某个节点比较慢的时候，系统会启动一个额外的节点作为其备份，形成某种竞争关系，哪个节点先结束就采用哪个节点的结果。在聚合梯度时，仅聚合一定比例的梯度，防止计算很慢的节点拖累聚合的效率。
  * 异步ADMM算法：允许部分速度较慢的workload暂时不参与当前的全局变量更新，当它的更新最终到达PS时，可以和其他workload的更新一起参与下一轮的全局变量更新。
  * 去中心化方法：让每个workload有更多的自主性，使模型的维护和更新更加分散化，易于扩展。每个节点可根据自己的需求来选择性地仅与少数其他节点通信。

### 基于模型集成的聚合方法
> 在凸优化问题中，average model的performance不会低于原有各个模型性能的平均值。但对于非凸的DNN，则不再适用，为了解决该问题，提出了基于模型集成的聚合方法。

#### 基于输出加和的聚合
虽然DNN的Loss function关于模型参数是非凸的，但是它关于模型的输出一般是非凸。则有：
$$
l(\frac{1}{K}\sum_{k=1}^K g(w^k; x), y)\leq \frac{1}{K}\sum_{k=1}^K l(g(w^k;x),y)
$$
因此，**若对局部模型的输出进行加和或平均，所得到的的预测结果要好于局部模型预测的平均值**。

但若简单集成$K$个模型，则最终模型的参数会增大$K$倍，“集成-压缩”算法就是为了解决该问题：

1. 各个workload依照本地局部数据训练出局部模型
2. 工作节点之间的相互通信获取彼此的局部模型，集成所有的局部模型得到集成模型，并对局部数据使用集成模型进行再标注
3. 利用模型压缩技术，结合数据的再标注信息，在每个工作节点上分别进行模型压缩，获得与局部模型大小相同的新模型作为最终的局和结果

模型集成与模型平均相比，最大的差别在于引入了**模型压缩**，常见的模型压缩有``Knowledge Distillation``。

#### 基于投票的聚合
以``Decision Tree``为例，首先，利用每个workload的局部数据对最佳特征及其分割点进行预计算，然后将本地选出的前$a$个特征告知中心服务器。中心服务器把来自各个workload的最佳特征聚合在一起，进行一次投票，选出全局的前$a$个最佳特征。然后中心服务器会与各个workload进行一次通信，要求它们把与这$a$个特征对应的直方图信息汇总到中心服务器，根据直方图信息再进行一次计算，从而判断出这$a$个候选特征中哪一个才是真正的全局最佳特征，并且计算它的最佳分割点。



## References
1. Dean J, Ghemawat S. [MapReduce: simplified data processing on large clusters](https://static.googleusercontent.com/media/research.google.com/zh-CN//archive/mapreduce-osdi04.pdf)[J]. Communications of the ACM, 2008, 51(1): 107-113.
2. Abadi M, Barham P, Chen J, et al. [Tensorflow: A system for large-scale machine learning](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)[C]//12th {USENIX} symposium on operating systems design and implementation ({OSDI} 16). 2016: 265-283.
3. Paszke A, Gross S, Massa F, et al. [Pytorch: An imperative style, high-performance deep learning library](https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf)[C]//Advances in neural information processing systems. 2019: 8026-8037.
4. 铁岩，陈薇，王太峰，高飞. [分布式机器学习：算法、理论与实践](https://book.douban.com/subject/30360968/)[M]//机械工业出版社. 2018.
