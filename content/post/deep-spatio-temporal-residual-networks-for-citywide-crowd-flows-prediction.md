---
categories:
- 论文阅读笔记
date: 2019-03-08 10:26:16+0000
description: AAAI 2017, ST-ResNet，网格流量预测，用三个相同结构的残差卷积神经网络对近邻时间、周期、趋势（远期）分别建模。与 RNN
  相比，RNN 无法处理序列长度过大的序列。三组件的输出结果进行集成，然后和外部因素集成，得到预测结果。原文地址：[Deep Spatio-Temporal Residual
  Networks for Citywide Crowd Flows Prediction](https://arxiv.org/abs/1610.00081)
draft: false
math: true
tags:
- deep learning
- Spatial-temporal
- graph convolutional network
- Graph
- Time Series
title: Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction
---
AAAI 2017, ST-ResNet，网格流量预测，用三个相同结构的残差卷积神经网络对近邻时间、周期、趋势（远期）分别建模。与 RNN 相比，RNN 无法处理序列长度过大的序列。三组件的输出结果进行集成，然后和外部因素集成，得到预测结果。原文地址：[Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction](https://arxiv.org/abs/1610.00081)
<!--more-->

# Abstract

对于交通管理和公共安全来说，预测人流很重要，但这个问题也很有挑战性，因为收到很多复杂的因素影响，如区域内的交通、事件、天气。我们提出了一个基于深度学习的模型 ST-ResNet，对城市内的每个区域的人流的进出一起预测。我们基于时空数据独一的属性，设计了一个端到端的结构。我们使用残差神经网络框架对时间近邻、周期、趋势属性建模。对每个属性，我们设计了残差卷积的一个分支，每个分支对人流的空间属性建模。ST-ResNet 基于数据动态地聚合三个残差神经网络的输出，给不同的分支和区域分配权重。聚合结果还融合了外部因素，像天气或日期。实验在北京和纽约两个数据集上开展。

# Introduction

对于交通管理和公共安全来说，预测人流很重要（Zheng et al. 2014）。举个例子，2015年新年夜，上海有大量人群涌入一个区域，导致 36 人死亡。2016年六月中旬，数百名 Pokemon Go 玩家冲入纽约中央公园，为了抓一只特别稀有的怪，导致严重的踩踏事故。如果可以预测一个区域的人流，这样的悲剧可以通过应急措施避免，像提前做交通管控，发布预警，疏散人群等。

我们在这篇文章中预测两类人流（Zhang et al. 2016)：如图 1（a）所示，流入和流出。流入是在给定时间段，从其他区域进入到一个区域的交通运载量。流出表示给定时段内，从一个区域向其他区域的交通运载量。两个流量都是区域间的人口流动。了解这个对风险评估和交通管理有很大帮助。流入/流出可以通过行人数量、邻近道路车辆数、公共运输系统的人数、或是所有的都加起来。图 1（b）展示了一个例子。我们可以使用手机信号测量行人数，$r\_2$ 的流入和流出分别为 3 和 1。类似地，使用车辆 GPS 轨迹，分别是 0 和 3。

![Figure1](/images/deep-spatio-temporal-residual-networks-for-citywide-crowd-flows-prediction/Fig1.JPG)

然而，同时预测城市每个区域人口的流入和流出是很有难度的，有 3 个复杂的因素：
1. 空间依赖。区域 $r\_2$ 的流入（图1（a））受邻近区域（像 $r\_1$）和遥远区域流出的影响。$r\_2$ 的流出也受其他区域（$r\_3$）流入的影响。$r\_2$ 的流入也影响其自身。
2. 时间依赖。一个区域的人流受到近期和远期时间影响。举个例子，早上8点发生的交通拥堵可能会影响到 9 点。此外，早高峰的交通状况可能在接连的几天都是相似的，每 24 小时一次。而且随着冬天的到来，早高峰时间可能越来越晚。温度下降，日初变晚会使人们起床时间变晚。
3. 外部影响。一些像天气和事件的外部因素可能会显著地改变城市内不同区域的人口流动。

为了解决这些问题，我们提出了一个深度深空残差网络 (ST-ResNet) 对每个区域的流入和流出同时预测。我们的贡献有 4 点：

- ST-ResNet 使用基于卷积的残差神经网络对城市内两个邻近的和遥远的区域的空间依赖建模，同时还确信了模型的预测精度不会因为模型的深度增加而降低。
- 我们将人口流动的时间属性分为三种，时间近邻、周期、趋势。ST-ResNet 使用三个残差网络对这些属性建模。
- ST-ResNet 动态地聚合三个上述网络的输出，给不同的分支和区域分配权重。聚合还融合了外部因素。
- 我们使用北京出租车的轨迹数据和气象数据，纽约自行车轨迹数据。结果表示我们的方法比 6 个 baseline 都好。

# Preliminaries

简要回顾人流预测问题（Zhang el al. 2016; Hoang, Zheng, and Singh 2016），介绍残差学习（He et al. 2016）。

## Formulation of Crowd Flows Problem

**Definition 1(Region (Zhang et al. 2016))** 根据不同粒度级和语义，一个地点的定义有很多。我们根据经纬度将城市划分成 $I \times J$ 个网格，一个网格表示一个区域，如图 2(a)。

![Figure2](/images/deep-spatio-temporal-residual-networks-for-citywide-crowd-flows-prediction/Fig2.JPG)

**Definition 2(Inflow/outflow (Zhang et al. 2016)) $\mathbb{P}$ 是第 t 时段的轨迹集合。对于第 $i$ 行第 $j$ 列的网格，时段 $t$ 流入和流出的人流分别定义为：

$$
x^{in,i,j}\_t = \sum\_{T\_r \in \mathbb{P}} \vert \lbrace k > 1 \mid g\_{k-1} \not \in (i, j) \wedge g\_k \in (i,j) \rbrace \vert
\\
x^{out,i,j}\_t = \sum\_{T\_r \in \mathbb{P}} \vert \lbrace k \geq 1 \mid g\_k \in (i,j) \wedge g\_{k+1} \not \in (i,j) \rbrace \vert
$$

其中 $T\_r: g\_1 \rightarrow g\_2 \rightarrow \cdots \rightarrow g\_{\vert T\_r \vert}$ 是 $\mathbb{P}$ 中的轨迹，$g\_k$ 是地理坐标；$g\_k \in (i,j)$ 表示点 $g\_k$ 落在 $(i, j)$ 内；$\vert · \vert$ 表示集合基数。

时段 $t$ ，所有区域的流入和流出可以表示成 $\mathbf{X}\_t \in \mathbb{R}^{2 \times I \times J}$，$(\mathbf{X}\_t)\_{0,i,j}=x^{in,i,j}\_t, (\mathbf{X}\_t)\_{1,i,j} = x^{out,i,j}\_t$。流入矩阵如图2(b)。

空间区域可以表达成一个 $I \times J$ 的区域，有两类流动，所以观测值可以表示为 $\mathbf{X} \in \mathbb{R}^{2 \times I \times J}$。

**Problem 1** 给定历史观测值 $\lbrace \mathbf{X}\_t \mid t = 0,\dots,n-1 \rbrace$，预测 $\mathbf{X}\_n$。

## Deep Residual Learning

$$\tag{1}
\mathbf{X}^{(l+1)} = \mathbf{X}^{(l)} + \mathcal{F}(\mathbf{X}^{(l)})
$$

# Deep Spatio-Temporal Residual Networks

![Figure3](/images/deep-spatio-temporal-residual-networks-for-citywide-crowd-flows-prediction/Fig3.JPG)

图 3 展示了 ST-ResNet的架构，4 个部分分别对时间近邻、周期、远期、外部因素建模。如图 3 所示，首先将流入和流出作为两个通道放到矩阵中，使用定义 1 和 2 引入的方法。我们将时间轴分为三个部分，表示近期时间、邻近历史、远期历史。三个时段的两通道的流动矩阵分别输入上述模型，对三种时间属性建模。这三个组件结构相同，都是残差网络。这样的结构捕获邻近和遥远区域间的空间依赖。外部组件中，我们手动的从数据集中提取了特征，如天气、事件等，放入两层全连接神经网络中。前三个组件的输出基于参数矩阵融合为 $\mathbf{X}\_{Res}$，参数矩阵给不同的区域不同的组件分配权重。$\mathbf{X}\_{Res}$ 然后与外部组件 $\mathbf{X}\_{Ext}$ 集成。最后，聚合结果通过 Tanh 映射到 $[-1, 1]$，在反向传播会比 logistic function 收敛的更快 (LeCun et al. 2012)。

## Structures of the First Three Components

如图 4。

![Figure4](/images/deep-spatio-temporal-residual-networks-for-citywide-crowd-flows-prediction/Fig4.JPG)

***Convolution*** 一个城市通常很大，包含很多距离不同的区域。直观上来说，邻近区域的人流会影响其他区域，可以通过 CNN 有效地处理，CNN 也被证明在层级地捕获空间信息方面很强 (LeCun et al. 1998)。而且，如果两个遥远地方通过地铁或高速公路连接，那么这两个区域间就有依赖关系。为了捕获任何区域的空间依赖，我们需要设计一个很多层的 CNN 模型，因为一个卷积层只考虑空间近邻，受限于它卷积核的大小。同样的问题在视频序列生成任务中也有，当输入和输出有同样的分辨率的时候(Mathieu, Couprie, and LeCun 2015)。为了避免下采样导致的分辨率损失引入了几种方法，同时还保持遥远的依赖关系(Long, Shelhamer, and Darrell 2015)。与传统的 CNN 不同的是，我们没有使用下采样，而是只使用卷积 (Jain et al. 2007)。如图 4(a)，图中有 3 个多级的 feature map，通过一些卷积操作相连。一个高层次的结点依赖于 9 个中间层次的结点，这些又依赖于低层次的所有结点。这意味着一个卷积可以很自然地捕获空间近邻依赖，堆叠卷积可以更多地捕获遥远的空间依赖。

图 3 的近邻组件使用了一些 2 通道流动矩阵对近邻时间依赖建模。令最近的部分为 $[\mathbf{X}\_{t-l\_c}, \mathbf{X}\_{t-(l\_c-1)}, \dots, \mathbf{X}\_{t-1}]$，也称为近邻依赖序列。我们将他们沿第一个轴（时间）拼接，得到一个张量 $\mathbf{X}^{(0)}\_c \in \mathbb{R}^{2l\_c \times I \times J}$，然后使用卷积（图 3 中的 Conv1）：

$$\tag{2}
\mathbf{X}^{(1)}\_c = f(W^{(1)}\_c \ast \mathbf{X}^{(0)}\_c + b^{(1)}\_c)
$$

其中 $\ast$ 表示卷积；$f$ 是激活函数；$W^{(1)}\_c, b^{(1)}\_c$ 是参数。

***Residual Unit.*** 尽管有 ReLU 职业那个的激活函数和正则化技巧，深度卷积网络在训练上还是很难。但我们仍然需要深度神经网络捕获非常大范围的依赖。对于典型的流量数据，假设输入大小是 $32 \times 32$，卷积核大小是 $3 \times 3$，如果我们想对城市范围的依赖建模，至少需要连续 15 个卷积层。为了解决这个问题，我们使用残差学习(He et al. 2015)，在训练超过 1000 层的网络时很有效。

在我们的 ST-ResNet(如图 3)，我们在 Conv1 上堆叠 $L$ 个残差单元如下：

$$\tag{3}
\mathbf{X}^{(l+1)}\_c = \mathbf{X}^{(l)}\_c + \mathcal{F}(\mathbf{X}^{(l)}\_c; \theta^{(l)}\_c), l = 1, \dots, L
$$

$\mathcal{F}$ 是残差函数，即 ReLU + Convolution，如图 4(b)。我们还在 ReLU 之前加了 *Batch Normalization*。在第 $L$ 个残差单元前，我们使用了一个卷积层，图 3 中的 Conv2。2 个卷积和 $L$ 个残差单元，图 3 中的近邻组件的输出是 $\mathbf{X}^{(l+2)}\_c$。

同样的，使用上面的操作，我们可以构建 *周期* 和 *趋势* 组件，如图 3。假设时段 $p$ 有 $l\_p$ 个时间间隔。那么 *时段* 依赖序列是 $[\mathbf{X}\_{t-l\_p \cdot p}, \mathbf{X}\_{t-(l\_p - 1) \cdot p}, \dots, \mathbf{X}\_{t-p}]$。使用式 2 和 式 3 那样的卷积和 $L$ 个残差单元，*周期* 组件的输出是 $\mathbf{X}^{(L + 2)}\_p$。同时，*趋势* 组件的输出是 $\mathbf{X}^{(L+2)}\_q$，输入是 $[\mathbf{X}\_{t-l\_q \cdot q}, \mathbf{X}\_{t-(l\_q - 1) \cdot q}, \dots, \mathbf{X}\_{t-q}]$，$l\_q$ 是*趋势*依赖序列的长度，$q$ 是趋势跨度。需要注意的是 $p$ 和 $q$ 是两个不同类型的周期。在实际的实现中，$p$ 等于一天，描述的是日周期，$q$ 是一周，表示周级别的趋势。

## The Structure of the External Component

交通流会被很多复杂的外部因素所影响，如天气或事件。图 5(a) 表示假期（春节）时的人流和平时的人流很不一样。图 5(b) 表示相比上周的同一天，突然而来的大雨会减少此时办公区域的人流。令 $E\_t$ 为特征向量，表示预测的时段 $t$ 的外部因素。我们的实现中，我们主要考虑天气、假期事件、元数据（工作日、周末）。详细情况见表 1。为了预测时段 $t$ 的交通流，假期事件和元数据可以直接获得。然而，未来时段 $t$ 的天气预报不知道。可以使用时段 $t$ 的天气预报，或是 $t-1$ 时段的天气来近似。我们在 $E\_t$ 上堆叠两个全连接层，第一层可以看作是每个子因素的嵌入层。第二层用来从低维映射到和 $\mathbf{X}\_t$ 一样的高维上。图 3 中外部组件的输出表示为 $\mathbf{X}\_{Ext}$，参数是$\theta\_{Ext}$。

![Figure5](/images/deep-spatio-temporal-residual-networks-for-citywide-crowd-flows-prediction/Fig5.JPG)

## Fusion

我们先用一个参数矩阵融合前三个组件，然后融合外部组件。

图6(a)和(d)展示了表1展示的北京轨迹数据的比例曲线，x轴是两个时段的时间差，y轴是任意两个有相同时间差的流入的平均比例。两个不同区域的曲线在时间序列上表现出了时间联系，也就是近期的流入比远期的流入更相关，表现出了事件近邻性。两条曲线有两个不同的形状，表现出不同区域可能有不同性质的近邻性。图6(b)和(e)描绘了7天所有时段的流入。我们可以观察到两个区域明显的日周期性。在办公区域，工作日的峰值比周末的高很多。住宅区在工作日和周末有相似的峰值。图6(c)和(f)描述了2015年3月到2015年6月一个特定时段(9:00pm-9:30pm)的流入。随着时间的推移，办公区域的流入逐渐减少，住宅区逐渐增加。不同的区域表现出了不同的趋势。总的来说，两个区域的流入受到近邻、周期、趋势三部分影响，但是影响程度是不同的。我们也发现其他区域也有同样的性质。

![Figure6](/images/deep-spatio-temporal-residual-networks-for-citywide-crowd-flows-prediction/Fig6.JPG)

综上，不同区域受近邻、周期、趋势的影响，但是影响程度不同。受这些观察的启发，我们提出了一个基于矩阵参数的融合方法。

***Parametric-matrix-based fusion.*** 我们融合图 3 中前三个组件：

$$\tag{4}
\mathbf{X}\_{Res} = \mathbf{W}\_c \odot \mathbf{X}^{(L+2)}\_c + \mathbf{W}\_p \odot \mathbf{X}^{(L+2)}\_p + \mathbf{W}\_q \odot \mathbf{X}^{(L+2)}\_q
$$

$\odot$ 是哈达玛乘积，$\mathbf{W}$ 是参数，分别调整三个组件的影响程度。

***Fusing the external component.*** 我们直接地将前三个组件的输出和外部组件融合，如图3。最后，时段 $t$ 的预测值，表示为 $\hat{\mathbf{X}}\_t$ 定义为：

$$\tag{5}
\hat{\mathbf{X}}\_t = \mathrm{tanh}(\mathbf{X}\_{Res} + \mathbf{X}\_{Ext})
$$

我们的 ST-ResNet 可以从三个流动与朕和外部因素特征通过最下滑 MSE 来训练：

$$\tag{6}
\mathcal{L}(\theta) = \Vert \mathbf{X}\_t - \hat{\mathbf{X}}\_t \Vert^2\_2
$$

## Algorithm and Optimization

算法1描述了 ST-ResNet 的训练过程。首先从原始序列构造训练实例。然后通过反向传播，用 Adam 算法训练。

![Algorithm1](/images/deep-spatio-temporal-residual-networks-for-citywide-crowd-flows-prediction/Alg1.JPG)

# Experiments

## Settings

**Datasets.** 我们使用表 1 中展示的两个数据集。每个数据集都包含两个子集，轨迹和天气。

- TaxiBJ: 轨迹数据是出租车 GPS 数据和北京的气象数据，2013年7月1日到10月30日，2014年5月1日到6月30日，2015年5月1日到6月30日，2015年11月1日到2016年4月1日。使用定义2，我们获得两类人流。我们选择最后四周作为测试集，之前的都为训练集。
- BikeNYC: 轨迹数据是2014年NYC Bike系统中取的，从4月1日到9月30日。旅行数据包含：持续时间、起点终点站点ID，起始终止时间。在数据中，最后10天选做测试集，其他选做训练集。

![Table1](/images/deep-spatio-temporal-residual-networks-for-citywide-crowd-flows-prediction/Table1.JPG)

**Baselines.** 我们对比了6个baselines：HA, ARIMA, SARIMA, VAR, ST-ANN, DeepST.

![Table2](/images/deep-spatio-temporal-residual-networks-for-citywide-crowd-flows-prediction/Table2.JPG)

![Table3](/images/deep-spatio-temporal-residual-networks-for-citywide-crowd-flows-prediction/Table3.JPG)