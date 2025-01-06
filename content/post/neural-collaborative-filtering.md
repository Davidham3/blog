---
categories:
- 论文阅读笔记
date: 2020-05-13 17:07:14+0000
description: 《Neural Collaborative Filtering》WWW 2017。这篇论文使用全连接神经网络实现了一个矩阵分解的推荐系统。给定一个user的特征表示，给定一个item的特征表示，通过神经网络输出用户对这个item感兴趣的分数。原文链接：[Neural
  Collaborative Filtering](https://arxiv.org/abs/1708.05031)。
draft: false
math: true
tags:
- deep learning
- recommender system
title: Neural Collaborative Filtering
---
《Neural Collaborative Filtering》WWW 2017。这篇论文使用全连接神经网络实现了一个矩阵分解的推荐系统。给定一个user的特征表示，给定一个item的特征表示，通过神经网络输出用户对这个item感兴趣的分数。原文链接：[Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)。
<!--more-->

最近看了看推荐系统，这篇论文提出了一个 NeuMF 的模型，基于全连接层的一个模型。

模型很简单，主要分两个部分，一个GMF部分，一个MLP部分。

# Generalized Matrix Factorization (GMF)

这个部分是 user 的特征向量与 item 的特征向量做 element-wise product。

# Multi-Layer Perceptron (MLP)

多层感知机部分就是将 user 和 item 的特征向量拼接，然后放到多层全连接里面。

# NeuMF

最后将这两个模块的输出拼接，然后丢到一个全连接层里面，映射到目标的分数，0到1，然后使用对数损失，训练即可。

# Experiments

实验部分，数据集的划分是，拿到用户所有的正例后，对每个用户，随机取一个作为测试集要预测的正例。然后取一定数量的负样本作为测试集的负例。

其他的正例作为训练集的正例，训练的时候通过采样的方式取负样本丢到网络中训练。

评价指标有两个：Hit Ratio 和 Normalized Discounted Cumulative Gain。

测试的时候，我们上述的“一定数量的负样本作为测试集的负例”，这个“一定数量”取999，那么测试集里面每个用户就有1000个样本，因为还要加那一个正例。让模型对这1000个样本进行预测，预测出1000个分数，取 top10。如果这最大的10个分数对应的 item 中，有那个正例，说明我们的模型在1000个 item 中，预测出的前十名里面成功命中那个正例了，那这个用户的 hit rate 就为1，否则为0，然后算所有用户的平均值即可，就是 hit ratio。

NDCG 的指标计算我还没有研究，等研究后再写。模型的代码看了 mxnet 官方提供的 example：[NeuMF](https://github.com/apache/incubator-mxnet/tree/master/example/neural_collaborative_filtering)，速度很快。

具体的实验结果，我在我自己用爬虫抓的一个数据集上看，效果没有 implicit 这个库里面的 als 效果好。ALS 的 HR@10 能跑到51%，我自己实现的 NeuMF 只能跑到48%，我使用的是 user 和 item 的 id 作为特征。但是 NeuMF 相比 ALS 的一个优势是可以加入 user 和 item 的其他特征信息，效果可能会更好一点，还需要进一步实验论证。