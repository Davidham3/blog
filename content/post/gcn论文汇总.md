---
categories:
- 论文阅读笔记
date: 2018-07-23 09:29:28+0000
draft: false
math: null
tags:
- Graph
title: GCN论文汇总
---
对看过的图神经网络做个总结，目前主要是GCN。
<!--more-->

Semi-Supervised Classification With Graph Convolutional Networks. Kipf & Welling 2017  
ICLR 2017。使用切比雪夫多项式的1阶近似完成了高效的图卷积架构。

优点|缺点
:-:|:-:
1阶近似，比k阶近似高效|卷积需使用整个图的拉普拉斯矩阵，图不能扩展

Convolution on Graph: A High-Order and Adaptive Approach.  
NIPS 2016，重新定义了卷积的定义，利用k阶邻接矩阵，定义考虑k阶邻居的卷积，利用邻接矩阵和特征矩阵构建能同时考虑顶点特征和图结构信息的卷积核。在预测顶点、预测图、生成图三个任务上验证了模型的效果。

优点|缺点
:-:|:-:


Graph Convolutional Neural Networks for Web-Scale Recommender Systems.  
KDD 2018。使用图卷积对顶点进行表示，学习顶点的embedding，通过卷积将该顶点的邻居信息融入到向量中。

优点|缺点
:-:|:-:
超大规模的图|

Diffusion-Convolutional Neural Networks.  
NIPS 2016。在卷积操作中融入了h-hop转移概率矩阵，通过对每个顶点计算该顶点到其他所有顶点的转移概率与特征矩阵的乘积，构造顶点新的特征表示，即diffusion-convolutional representation，表征顶点信息的扩散，然后乘以权重矩阵W，加激活函数，得到卷积的定义。在顶点分类和图分类上做了测试。

优点|缺点
:-:|:-:
没有增加模型的复杂度|空间复杂度高
使用转移概率矩阵|模型不能捕获尺度较大的空间依赖关系
| |不同的分类任务（顶点、图）有不同的卷积表达式

Graph Attention Networks.  
ICLR 2018。图注意力网络，使用self-attention来构建graph attentional layer，attention会考虑当前顶点所有的邻居对它的重要度，基于谱理论的模型不能应用到其他不同结构的图上，而这个基于attention的方法能有效的解决这个问题。

Inductive Representation Learning on Large Graphs.  
NIPS 2017。提出的方法叫GraphSAGE，针对的问题是之前的NRL是transductive，作者提出的GraphSAGE是inductive。主要考虑了如何聚合顶点的邻居信息，对顶点或图进行分类。

应用：

Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic.  
IJCAI 2018，大体思路：使用Kipf & Welling 2017的近似谱图卷积得到的图卷积作为空间上的卷积操作，时间上使用一维卷积对所有顶点进行卷积，两者交替进行，组成了时空卷积块，在加州PeMS和北京市的两个数据集上做了验证。

Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition.  
AAAI 2018，以人体关节为图的顶点，构建空间上的图，然后通过时间上的关系，连接连续帧上相同的关节，构成一个三维的时空图。针对每个顶点，对其邻居进行子集划分，每个子集乘以对应的权重向量，得到时空图上的卷积定义。实现时使用Kipf & Welling 2017的方法实现。

优点|缺点
:-:|:-:
将空间和时间一体化|实现上仍是Kipf & Welling的方法