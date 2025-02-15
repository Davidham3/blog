---
date: 2024-12-25 08:01:34+0000
description: 讲一个在spark上做DBSCAN的案例，记录一下过程。
draft: false
math: true
tags:
- Machine Learning
- Spark
title: DBSCAN with Spark
---

讲一个在spark上做DBSCAN的案例，记录一下过程。

<!--more-->

背景：有一组实体，每个实体下面挂了很多经纬度数据，需要对每个实体运行密度聚类获得类中心。实际就是一个数据清洗的方法，获得密度聚类结果主类后，取主类对应的经纬度的中心点。这样就可以获得实体与经纬度之间的关系，实现了数据清洗。

实现的时候，因为scikit-learn提供了DBSCAN的工具，因此直接使用pyspark实现。可以直接将每个实体的经纬度数据通过spark聚合，然后定义UDF，在UDF里面做DBSCAN，然后将结果返回到spark的DataFrame里面，然后再通过一些代码找出主类即可。DBSCAN里面如果类的id是-1，说明这个类是噪声，所以只要看一下占绝对优势的类是不是-1，还有它的占比，设定一个阈值就可以实现清洗了。

这里需要注意的是数据倾斜的问题：不同实体的经纬度数据量不一样，有的多有的少。由于DBSCAN的时间复杂度是 $O(n^2)$，所以如果数据量太大是算不过来的，而且会拖累整个任务的运行。

解决这个问题需要做两点：
1. 对于每个实体，设定一个经纬度数量的上限，比如每个实体最多只能有1000个经纬度。如果这个实体的经纬度数量超过1000个，直接采样，采到1000个。
2. 然后就是任务均分：因为肯定有大量的实体它的经纬度个数是不到1000个的，有可能有80%的实体的经纬度数都很少。如果分区的逻辑不对，可能会使得超过1000个的实体都聚集在1个分区里面，那这个分区肯定算的是最慢的，所以需要一个合理的分区方案。

分区方案很简单：
1. 设定一个分区数，比如200。
2. 构建一个数据结构KeySet，里面有一个List[str]，一个int，前者用来存储实体的名称，后者用来存储当前这些实体需要的计算次数。
3. 构建一个小顶堆，里面放置200（和分区数一样）个上面的KeySet。小顶堆通过KeySet的int值进行排序。所以小顶堆的堆顶一定是计算次数最小的KeySet。
4. 遍历所有的实体名称与他们的经纬度个数，把经纬度个数算一个平方，形成二元组(实体名称, 计算次数)。每次从小顶堆取出一个KeySet，然后将二元组插入这个KeySet，也就是实体名称插入list，计算次数加到int值上面。然后再将这个KeySet插入到堆里面。
5. 结束后可以获得一个200个KeySet的堆，每个KeySet里面有一个List[str]是实体名称，还有一个int值表示这些实体名称的总计算次数。
6. 将上面这个计算好的数据进行变换，形成一个map，key是实体名称，值是它的id，id就是0到199，随便赋值就可以了。
7. 把这个map广播到所有机器上，然后在dataframe里面通过这个map新增一列partition id，然后再通过repartition对这一列进行分区就好了。

这样就可以让每个分区里面的计算量大体相近了。这个算法是一个贪心的算法，最后拿到的结果不一定是最优的。如果想要最优解还需要其他的算法。

这个问题实际上是给定一组数字List[int]，给定200个桶，将这些数字放入这200个桶之后，将每个桶里面的数字相加，得到200个数字。使得这200个数字的标准差最小。