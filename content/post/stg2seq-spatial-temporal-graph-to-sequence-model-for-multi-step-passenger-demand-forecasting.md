---
categories:
- 论文阅读笔记
date: 2019-07-12 19:57:39+0000
description: 'IJCAI 2019. 原文链接：[STG2Seq: Spatial-temporal Graph to Sequence Model
  for Multi-step Passenger  Demand Forecasting](https://arxiv.org/abs/1905.10069.pdf)'
draft: false
math: true
tags:
- Graph
- graph convolutional network
- Spatial-temporal
- deep learning
title: 'STG2Seq: Spatial-temporal Graph to Sequence Model for Multi-step Passenger
  Demand Forecasting'
---
IJCAI 2019. 原文链接：[STG2Seq: Spatial-temporal Graph to Sequence Model for Multi-step Passenger
  Demand Forecasting](https://arxiv.org/abs/1905.10069.pdf)
<!--more-->

# Abstract

多步乘客需求预测对于按需车辆共享服务来说是个重要的任务。然而，预测多个时刻的乘客需求由于时空依赖的非线性和动态性很有挑战。我们提出了基于图的城市范围的旅客需求预测模型，使用一个层次图卷积同时捕获空间和时间关联性。我们的模型有三部分：1) 长期编码器对历史旅客需求编码；2) 短期编码器推导下一步预测结果来生成多步预测；3) 使用一个基于注意力的输出模块对动态的时间和各通道信息建模。实验在三个数据集上表明我们的方法比很多方法好。

# 1. Introduction

# 2. Notations and Problem Statement

假设一个城市分成 $N$ 个小的区域，不考虑是分成网格还是路网。我们将区域的集合表示为 $\lbrace r\_1, r\_2, \dots, r\_i, \dots r\_N \rbrace$。在每个时间步 $t$，一个二维矩阵 $\boldsymbol{D\_t} \in \mathbb{R}^{N \times d\_{in}}$ 表示所有区域在时间 $t$ 的旅客需求。另一个向量 $\boldsymbol{E\_t} \in \mathbb{R}^{d\_e}$ 表示时间步 $t$ 的时间特征，包含了几点、星期几以及节假日的信息。

给定城市范围的历史旅客需求序列 $\lbrace \bm{D\_0}, \bm{D\_1}, \dots, \bm{D\_t} \rbrace$ 和时间特征 $\lbrace \bm{E\_0}, \bm{E\_1}, \dots, \bm{E\_{t+\tau}} \rbrace$，目标是学习一个预测函数 $\Gamma(\cdot)$ 来预测接下来的 $\tau$ 个时间步上城市范围的旅客需求序列。我们只使用历史 $h$ 个时间步的需求序列作为输入 $\lbrace \bm{D\_{t-h+1}, \bm{D\_{t-h+2}}, \dots, \bm{D\_t}} \rbrace$。我们的任务描述为：

$$\tag{1}
(\bm{D\_{t+1}}, \bm{D\_{t+2}}, \dots, \bm{D\_{t+\tau}}) = \Gamma(\bm{D\_{t-h+1}}, \bm{D\_{t-h+2}}, \dots, \bm{D\_t}; \bm{E\_0}, \bm{E\_1}, \dots, \bm{E\_{t+\tau}})
$$

# 3. Methodology

STG2Seq 的架构有三个组件：1. 长期编码器，2. 短期编码器，3.基于注意力的输出模块。长期和短期编码器由多个序列时空门控图卷积模块 (GGCM) 组成，通过在时间维度使用 GCN 可以同时捕获时间和空间相关性。

![Figure2](/blog/images/stg2seq-spatial-temporal-graph-to-sequence-model-for-multi-step-passenger-demand-forecasting/Fig2.JPG)

## 3.1 Passenger Demand on Graph

我们先介绍如何将城市范围的旅客需求在图上描述出来。之前的工作假设一个区域的旅客需求会被近邻的区域影响。然而，我们认为空间关系并不是仅依赖空间位置。如果遥远的区域和当前区域有相似的地方，比如具有相似的 POI，那么也可能拥有相同的旅客需求模式。因此，我们将城市看作一个图 $G = (v, \xi, A)$，$v$ 是区域的集合 $v = \lbrace r\_i \mid i=1,2,\dots,N \rbrace$，$\xi$ 表示边的集合，$A$ 是邻接矩阵。我们根据区域间旅客需求模式的相似性定义图的边。

$$\tag{2}
A\_{i, j} = \begin{cases}
1, \quad \text{if} \quad Similarity\_{r\_i, r\_j} > \epsilon\\
0, \quad \text{otherwise}
\end{cases}
$$

其中 $\epsilon$ 是阈值，控制 $A$ 的稀疏程度。为了定量区域间的旅客需求模式的相似性，我们使用皮尔逊相关系数。$D\_{0\text{\textasciitilde}t}(r\_i)$ 表示时间从 0 到 $t$ 的区域 $r\_i$ 历史旅客需求序列。$r\_i$ 和 $r\_j$ 之间的相似度可以定义为：

$$\tag{3}
Similarity\_{r\_i, r\_j} = Pearson(D\_{0\text{\textasciitilde}t}(r\_i), D\_{0\text{\textasciitilde}t}(r\_j))
$$

## 3.2 Long-term and Short-term Encoders

很多之前的工作只考虑下一步预测，即预测下一时间步的旅客需求。在训练过程中通过减少下一时间步预测值的误差而不考虑后续时间步的误差来优化模型。因此，这些方法在多步预测的问题上会退化。仅有一些工作考虑了多步预测的问题 [Xingjian et al., 2015; Li et al., 2018]。这些工作采用了基于 RNN 的编码解码器的架构，或是它的变体，比如 ConvLSTM 这样的作为编码解码器。这些方法有两个劣势：1. 链状结构的 RNN 在编码的时候需要遍历输入的时间步。因此他们需要与输入序列等长的 RNN 单元个数（序列多长，RNN单元就有多少个）。在目标需求和前一个需求上的长距离计算会导致一些信息的遗忘。2. 在解码部分，为了预测时间步 $T$ 的需求，RNN 将隐藏状态和前一时间步 $T-1$ 作为输入。因此，前一时间步带来的误差会直接影响到预测，导致未来时间步误差的累积。

不同于之前所有的工作，我们引入了一个依赖于同时使用长期和短期编码器的架构，不用 RNN 做多步预测。长期编码器取最近的 $h$ 个时间步的城市历史旅客需求序列 $\lbrace \bm{D\_{t-h+1}}, \bm{D\_{t-h+2}}, \dots, \bm{D\_t} \rbrace$ 作为输入来学习历史的时空模式。这 $h$ 步需求合并后组织成三维矩阵，$h \times N \times d\_{in}$。长期编码器由一些 GGCM 组成，每个 GCGGM 捕获在所有的 $N$ 个区域上捕获空间关联性，在 $k$ 个时间步上捕获时间关联性。$k$ 是超参数，我们会在 3.3 节讨论。因此，只需要 $\frac{h-1}{k-1}$ 个迭代的步数就可以捕获 $h$ 个时间步上的时间关联性。对比 RNN 结构，我们的基于 GGCM 的长期编码器显著的降低了遍历长度，进一步减少了信息的损失。长期编码器的输出 $Y\_h$ 的维数是 $h \times N \times d\_{out}$，是输入的编码表示。

短期编码器用来集成已经预测的需求，用于多步预测。它使用一个长度为 $q$ 的滑动窗来捕获近期的时空关联性。当预测在 $T(T \in [t+1,t+\tau])$ 步的旅客需求时，它取最近的 $q$ 个时间步的旅客需求，即 $\lbrace \bm{D\_{T-q}}, \bm{D\_{T-q+1}}, \dots, \bm{D\_{T-1}} \rbrace$ 作为输入。除了时间步的长度以外，短期编码器和长期编码器一样。短期编码器生成一个维数为 $q \times N \times d\_{out}$ 的矩阵 $Y^T\_q$ 作为近期趋势表示。和基于 RNN 的解码器不同的是，RNN的解码器只将最后一个时间步的预测结果输入回去。因此，预测误差会被长期编码器小柔，减轻基于 RNN 的解码器会导致误差累积的问题。

## 3.3 Gated Graph Convolutional Module

门控图卷积模块是长期编码器和短期编码器的核心。每个 GGCM 由几个 GCN 层组成，沿着时间轴并行。为了捕获时空关联性，每个 GCN 在一定长度的时间窗内操作($k$)。它可以提取 $k$ 个时间步内所有区域的空间关联性。通过堆叠多个 GGCM，我们的模型形成了一个层次结构，可以捕获整个输入的时空关联性。图 3 展示了只使用 GCN 捕获时空关联性，为了简化我们忽略了通道维。Yu et al., 2018 的工作和我们的 GGCM 模块很像。他们的工作首先使用 CNN 捕获时间关联性，然后使用 GCN 捕获空间关联性。我们的方法对比他们的方法极大的简化了，因为我们可以同时捕获时空关联性。

![Figure3](/blog/images/stg2seq-spatial-temporal-graph-to-sequence-model-for-multi-step-passenger-demand-forecasting/Fig3.JPG)

![Figure4](/blog/images/stg2seq-spatial-temporal-graph-to-sequence-model-for-multi-step-passenger-demand-forecasting/Fig4.JPG)

GGCM 模块的详细设计如图 4。第 $l$ 个 GGCM 的输入是一个矩阵，维数为 $h \times N \times C^l$。在第一个 GGCM 模块，$C^l$ 是 $d\_{in}$ 维的。第 $l$ 个 GGCM 的输出是 $h \times N \times C^{l+1}$。我们先拼接一个 zero padding，维数为 $(k-1) \times N \times C^l$，得到新的输入 $(h+k-1) \times N \times C^l$，确保变换不会减少序列的长度。接下来，GGCM 中的每个 GCN 取 $k$ 个时间步的数据 $k \times N \times C^l$ 作为输入来提取时空关联性，然后 reshape 成一个二维矩阵 $N \times (k \cdot C^l)$。根据 Kipf & Welling 的 GCN，GCN 层可以描述如下：

$$\tag{4}
X^{l+1} = (\tilde{P}^{-\frac{1}{2}} \tilde{A} \tilde{P}^{-\frac{1}{2}}) X^l W
$$

$\tilde{A} = A + I\_n$，$\tilde{P}\_{ii} = \sum\_j \tilde{A}\_{ij}$，$X \in \mathbb{R}^{N \times (k \cdot C^l)}$，$W \in \mathbb{R}^{(k \cdot C^l) \times C^{l+1}}$，$X^{l+1} \in \mathbb{R}^{N \times C^{l+1}}$
。

除此以外，我们使用了门控机制对旅客需求预测的复杂非线性建模。式 4 重新描述如下：

$$\tag{5}
X^{l+1} = ((\tilde{P}^{-\frac{1}{2}} \tilde{A} \tilde{P}^{-\frac{1}{2}}) X^l W\_1 + X^l) \otimes \sigma((\tilde{P}^{-\frac{1}{2}} \tilde{A} \tilde{P}^{-\frac{1}{2}}) X^l W\_2)
$$

$\otimes$ 是对应元素相乘，$\sigma$是 sigmoid 激活函数。因此输出是一个非线性门 $\sigma((\tilde{P}^{-\frac{1}{2}} \tilde{A} \tilde{P}^{-\frac{1}{2}}) X^l W\_2)$ 控制的线性变换 $((\tilde{P}^{-\frac{1}{2}} \tilde{A} \tilde{P}^{-\frac{1}{2}}) X^l W\_1 + X^l)$。非线性门控制线性变换的哪个部分可以通过门影响预测。此外，我们使用残差连接来避免式 5 中的网络退化。

最后，门控机制产生的 $h$ 个输出沿时间轴合并，生成 GGCM 模块的输出 $h \times N \times C^{l+1}$。

## 3.4 Attention-based Output Module

如 3.2 描述的那样，长期时空依赖和 $T$ 时间步的近期时空依赖通过两个矩阵描述 $Y\_h$ 和 $Y^T\_q$。我们拼接。我们拼接他们形成联合表示 $Y\_{h+q} \in \mathbb{R}^{(h+q) \times N \times d\_{out}}$，通过一个基于注意力机制的模块解码获得预测值。这里为了简便忽略 $T$。$Y\_{h+q}$ 的三个轴分别是时间、空间、通道。

我们先引入一个时间注意力机制来解码 $Y\_{h+q}$。旅客需求是一个典型的时间序列，前一时刻的需求对后一时刻有影响。然而，之前的每一步对预测目标的影响是不同的，影响随时间变化。我们设计了一个时间注意力机制对每个历史时间步增加注意力分数衡量其影响。分数通过 $Y\_{h+q} = [y\_1, y\_2, \dots, y\_{h+q}](y\_i \in \mathbb{R}^{N \times d\_{out}})$ 和目标时间步的时间特征 $\bm{E}\_T$ 生成，这个分数可以自适应地学习之前的时间步随时间的动态影响。我们定义时间注意力分数如下：

$$\tag{6}
\bm{\alpha} = softmax(tanh(Y\_{h+q} W^Y\_3 + E\_T W^E\_4 + b\_1))
$$

$W^Y\_3 \in \mathbb{R}^{(h+q) \times (N \times d\_{out}) \times 1}$，$W^E\_4 \in \mathbb{R}^{d\_e \times (h+q)}$，$b\_1 \in \mathbb{R}^{(h+q)}$。联合表示 $Y\_{h+q}$ 通过注意力分数 $\bm{\alpha}$ 转换：

$$\tag{7}
Y\_{\alpha} = \sum^{h+q}\_{i=1} \alpha^i y\_i \quad \in \mathbb{R}^{N \times d\_{out}}
$$

受到 [Chen et al., 2017] 的启发，每个通道的重要性是不同的，我们在时间注意力后面加了一个通道注意力模块来找到 $Y\_\alpha = [y\_1, y\_2, \dots, y\_{d\_{out}}]$ 中最重要的那个。计算如下：

$$\tag{8}
\bm\beta = softmax(tanh(Y\_\alpha W^Y\_5 + E\_T W^E\_6 + b\_2))
$$

$$\tag{9}
Y\_{\beta} = \sum^{d\_{out}}\_{i=1} \beta^i y\_i \quad \mathbb{R}^N
$$

其中，$W^Y\_5 \in \mathbb{R}^{d\_{out} \times N \times 1}$，$W^E\_6 \in \mathbb{R}^{d\_e \times d\_{out}}$；$\bm\beta \in \mathbb{R}^{d\_{out}}$ 是每个通道的注意力分数。当预测的维度是1时，$Y\_\beta$ 就是我们预测的旅客需求 $\bm{D'\_T}$。当预测维度是 2 时（预测起止需求），我们给每个通道计算注意力分数，将他们拼接起来得到最后的预测值。