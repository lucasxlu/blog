---
title: "[DB] MongoDB"
date: 2020-09-25 21:08:37
mathjax: true
tags:
- Database
- NoSQL
- MongoDB
catagories:
- Database
- NoSQL
- MongoDB
---
# MongoDB
## Introduction
MongoDB是一种典型的``NoSQL``数据库，与常用的``RDBMS（例如MySQL）``相比，MongoDB没有显式的shema定义，因此数据结构非常灵活。MongoDB的基本数据结构为BSON，是一种类JSON的结构，下表展示了MongoDB与MySQL的对比：

| MongoDB | MySQL |
| :---: | :---: |
| db | db |
| collection | table |
| document | record |
| field | attribute |

MongoDB GUI管理工具有[Robo 3T](https://robomongo.org/)和[MongoDBCompass](https://www.mongodb.com/products/compass)，前者是商业软件，后者是MongoDB自己开发的，提供了基本的导入导出、CRUD语法转换等功能，强烈推荐[MongoDBCompass](https://www.mongodb.com/products/compass)。

基本的CRUD操作在此不赘述，请参考[MongoDB官方文档](https://docs.mongodb.com/manual/crud/)即可。下面开启本文重点——MongoDB的副本&集群搭建。


## Build MongoDB Replica
首先介绍一下什么是replica，MongoDB的官方文档是这么说的：
> A replica set in MongoDB is a group of mongod processes that maintain the same data set. Replica sets provide redundancy and high availability, and are the basis for all production deployments. 

MongoDB复制是将数据同步在多个服务器的过程。复制提供了数据的冗余备份，并在多个服务器上存储数据副本，提高了数据的可用性，并可以保证数据的安全性。复制还允许您从硬件故障和服务中断中恢复数据。摘录一些MongoDB官方文档里比较有价值的信息吧，原文比较长，有需要的话还是建议阅读[原文](https://docs.mongodb.com/manual/replication/)。

``Replica set``是保存了相同数据集的mongod instances，**1个replica set包含多个data bearing nodes以及1个可选的arbiter node**，只有1个data bearing node能作为primary node，其他的都是secondary node。
> A replica set is a group of mongod instances that maintain the same data set. A replica set contains several data bearing nodes and optionally one arbiter node. Of the data bearing nodes, **one and only one member is deemed the primary node, while the other nodes are deemed secondary nodes.**

![replica-set-read-write-operations](https://docs.mongodb.com/manual/_images/replica-set-read-write-operations-primary.bakedsvg.svg)

Primary node接受所有读写操作，并将所有操作记录在``oplog``，secondary node与primary node进行通信后，再异步地执行``oplog``记录，来使其与primary node保持data consistency。
> The primary node receives all write operations. A replica set can have only one primary capable of confirming writes with ``{w: "majority"}`` write concern. The secondaries replicate the primary's oplog and apply the operations to their data sets such that the secondaries' data sets reflect the primary's data set. If the primary is unavailable, an eligible secondary will hold an election to elect itself the new primary.

![replica-set-primary-with-two-secondaries](https://docs.mongodb.com/manual/_images/replica-set-primary-with-two-secondaries.bakedsvg.svg)

In some circumstances (such as you have a primary and a secondary but cost constraints prohibit adding another secondary), you may choose to add a mongod instance to a replica set as an arbiter. An arbiter participates in elections but does not hold data (i.e. does not provide data redundancy).

![replica-set-primary-with-secondary-and-arbiter](https://docs.mongodb.com/manual/_images/replica-set-primary-with-secondary-and-arbiter.bakedsvg.svg)

An arbiter will always be an arbiter whereas a primary may step down and become a secondary and a secondary may become the primary during an election.

When a primary does not communicate with the other members of the set for more than the configured electionTimeoutMillis period (10 seconds by default), an eligible secondary calls for an election to nominate itself as the new primary. The cluster attempts to complete the election of a new primary and resume normal operations.

![replica-set-trigger-election](https://docs.mongodb.com/manual/_images/replica-set-trigger-election.bakedsvg.svg)

默认情况下，client通过primary node完成数据读写，但也可以通过显式指定直接从secondary node进行数据读取。**需要注意的是：因为MongoDB中，primary node与secondary node之间的数据同步是异步的，所以显式从secondary node读取的数据可能与primary node存在data inconsistency的问题**。
> By default, clients read from the primary, however, clients can specify a read preference to send read operations to secondaries. **Asynchronous replication to secondaries means that reads from secondaries may return data that does not reflect the state of the data on the primary.**

![replica-set-read-preference-secondary](https://docs.mongodb.com/manual/_images/replica-set-read-preference-secondary.bakedsvg.svg)

