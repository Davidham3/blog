---
date: 2022-05-19 14:54:43+0000
description: NIPS 2019, [Unsupervised Scalable Representation Learning for Multivariate
  Time Series](https://arxiv.org/abs/1901.10738)。T-loss。无监督多元时间序列表示模型。利用word2vec的负样本采样的思想学习时间序列的嵌入表示。代码：[UnsupervisedScalableRepresentationLearningTimeSeries](https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries)
draft: false
math: true
tags:
- deep learning
- time series
- learning representations
title: Unsupervised Scalable Representation Learning for Multivariate Time Series
---

NIPS 2019, [Unsupervised Scalable Representation Learning for Multivariate Time Series](https://arxiv.org/abs/1901.10738)。T-loss。无监督多元时间序列表示模型。利用word2vec的负样本采样的思想学习时间序列的嵌入表示。代码：[UnsupervisedScalableRepresentationLearningTimeSeries](https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries)

<!--more-->

# 3 Unsupervised Training

只想训练编码器，不想搞编码解码结构，因为计算量太大了。因此引入了一个针对时间序列的triplet loss。

目标是无监督的情况下相似的时间序列具有相似的表示。

![Figure1](/blog/images/unsupervised-scalable-representation-learning-for-multivariate-time-series/Fig1.jpg)

图1，给定时间序列$y\_i$，一个随机子序列$x^{\text{ref}}$。$x^{\text{ref}}$的表示要与$x^{\text{pos}}$相似，与另一个随机采样的子序列$x^{\text{neg}}$不相似。类比word2vec，$x^{\text{pos}}$是word，$x^{\text{ref}}$是context，$x^{\text{neg}}$是一个随机的词。

损失函数：

$$
\tag{1} -\log{( \sigma(f(x^{\text{ref}}, \theta)^\top f(x^{\text{pos}}, \theta)) )} - \sum^K\_{k=1} \log{( \sigma( -f(x^{\text{ref}}, \theta)^\top f(x^{\text{neg}}, \theta) ) )},
$$

$f(\cdot, \theta)$是一个神经网络，参数是$\theta$，$\sigma$是sigmoid激活函数。$K$是负样本的个数

![Algorithm_1](/blog/images/unsupervised-scalable-representation-learning-for-multivariate-time-series/Alg1.jpg)

# 4 Encoder Architecture

1.  必须从时间序列中提取相关信息
2.  训练和推断的时候时间和空间都要高效
3.  必须能接受变长的输入

使用exponentially dilated causal convolutions处理。相比RNN，这类网络可以在GPU上高效地并行计算。对比fully convolutions，可以捕获更长范围的信息。

即便LSTM解决了梯度消失和爆炸的问题，在这个梯度问题上仍然比不过卷积网络。

我们的模型堆叠了多个dilated causal convolutions，如图2a。每个数都是通过前面的数计算出来的，因此称为因果。图2a红色部分展示了输出值的计算路径。

卷积网络的输出会放入一个max pooling层，把所有时间信息聚合到一个固定大小的向量中。