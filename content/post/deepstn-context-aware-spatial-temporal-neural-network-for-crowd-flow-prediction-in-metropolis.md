---
categories:
- 论文阅读笔记
date: 2019-05-08 16:40:48+0000
description: 'AAAI 2019，网格流量预测，对比ST-ResNet，抛出三个问题，卷积捕获的空间范围小、人口流动和区域的功能相关、之前的融合机制不好。改了一下残差卷积，给
  POI 信息增加了时间维度，多组件的信息提前融合，减少了参数，稳定模型训练。原文链接：[DeepSTN+: Context-aware Spatial-Temporal
  Neural Network for Crowd Flow Prediction in Metropolis](https://github.com/FIBLAB/DeepSTN/blob/master/docs/5624_AAAI19_DeepSTN%2B_Camera_Ready.pdf)'
draft: false
math: true
tags:
- deep learning
- Spatial-temporal
- graph convolutional network
- Graph
- Time Series
title: 'DeepSTN+: Context-aware Spatial-Temporal Neural Network for Crowd Flow Prediction
  in Metropolis'
---
AAAI 2019，网格流量预测，对比ST-ResNet，抛出三个问题，卷积捕获的空间范围小、人口流动和区域的功能相关、之前的融合机制不好。改了一下残差卷积，给 POI 信息增加了时间维度，多组件的信息提前融合，减少了参数，稳定模型训练。原文链接：[DeepSTN+: Context-aware Spatial-Temporal Neural Network for Crowd Flow Prediction in Metropolis](https://github.com/FIBLAB/DeepSTN/blob/master/docs/5624_AAAI19_DeepSTN%2B_Camera_Ready.pdf)
<!--more-->

# Abstract

人口流量预测在城市规划、交通管控中的很多应用中都很重要。目的是预测流入和流出流量。我们提出了 DeepSTN+，一个基于深度学习的卷积模型，预测超大城市的人口流量。首先，DeepSTN+ 使用 *ConvPlus* 结构对大范围的空间依赖建模。此外，POI 分布和时间因素相融合来表达区域属性的影响，以此引入人口流动的先验知识。最后，我们提出了一个有效的融合机制来稳定训练过程，提升了结果。基于两个真实数据集的大量实验结果表明我们模型的先进性，和 state-of-the-art 比高了 8% ~ 13% 左右。

# Introduction

如图 1 所示，人口流量预测是在给定历史流量信息的前提下，预测城市内每个区域的流入和流出流量。最近，为了解决这个问题，基于深度学习的模型被相继提出，获得了很好的效果。Deep-ST 是第一个使用卷积网络捕获空间信息的模型。ST-ResNet 用卷积模块替换了卷积。通过融合金字塔型的 ConvGRU 模型和周期表示，Periodic-CRN 设计成了捕获人口流动周期性的模型。

![Figure1](/images/deepstn-context-aware-spatial-temporal-neural-network-for-crowd-flow-prediction-in-metropolis/Fig1.JPG)

这些方法仍然不够有效且不精确：

1) *不能捕获区域间的空间依赖。* 由于现代城市中高级的运输系统的存在，人们可以通过地铁或出租车在短时间内移动到很远的地方。因此，区域间的大范围空间依赖在人口移动中逐渐扮演重要的角色。现存的工作使用多层卷积网络来建模。然而，它们只能一步一步地捕获近邻的空间依赖，不能直接地捕获大范围的空间依赖。
2) *忽略了人口流动的区域功能的影响。* 人口移动是发生在物理世界中的，会直接受到区域属性的影响。举个例子，人们通常早上从家出发到公司，晚上回来。显然，区域的功能（属性）包含了关于人类移动的先验知识。然而，现存的解决方案没有考虑过区域的属性。
3) *冗余以及不稳定的神经网络结构。* ST-ResNet 利用了三个独立分支，每个分支都是残差卷积单元，用来处理不同的输入，在模型的结尾用一个线性操作融合三个输出。但是，最后的融合机制导致不同组件间的交互产生了缺陷，这个缺陷导致了网络内产生了无效的参数和不稳定的性质。

总结一下，模型应该考虑大范围的空间依赖，区域的影响，更有效的融合机制这三点因素。我们提出的 DeepSTN+ 解决了上述挑战。我们设计了一个 *ConvPlus* 结构直接地捕获大范围空间依赖。*ConvPlus* 放在残差单元前面作为一个全局特征提取器提取出区域间的全局特征。其次，我们设计了一个 *SemanticPlus* 结构来学习人口在区域间移动的先验知识。用静态的 POI 分布作为输入，*SemanticPlus* 利用时间因素给不同时间上不同的 POI 分配权重。最后，我们引入早融合和多尺度融合机制来减少训练参数，捕获不同级别特征间的复杂关系。这样，我们的系统可以对更复杂的空间关联性建模，获得更好的效果，我们的贡献有以下几点：

- 我们设计了一个新的残差单元，ResPlus 单元用来替换原始的残差单元。我们指出了典型的卷积模型不能有效地捕获大范围依赖。ResPlus 包含了一个 *ConvPlus* 结构，可以捕获人流间的大范围空间依赖。
- 我们设计了一个 *SemanticPlus* 结构来建模不同区域的影响，学习人口流动的先验知识。我们在模型头部使用早融合机制，在结尾使用多尺度融合机制，提升了模型的精度和稳定性。
- 我们在两个数据集上开展了大量的实验，对比了 5 个 baselines，结果显示我们的模型在预测人口流动的错误上减少了 8% ~ 13%。

# Preliminaries

这部分，我们首先介绍人口流量预测问题，简要回顾 ST-ResNet。

## Problem Formulation

**Definition 1 (Region (Zhang et al. 2016)) 为了表示城市的区域，我们基于经纬度将城市划分成 $H \times W$ 个区域，所有的网格有相同大小且表示一个区域。

**Definition 2 (Inflow/outflow (Zhang et al. 2016)) 为了表示城市内的人口流动，我们定义了区域 $(h, w)$ 在时段 $i$ 的流入和流出流量：

$$
x^{h,w,in}\_{i} = \sum\_{T\_{r\_k} \in \mathbb{P}} \vert \lbrace j > 1 \mid g\_{j-1} \not \in (h, w) \And g\_j \in (h, w) \rbrace \vert,\\

x^{h,w,out}\_{i} = \sum\_{T\_{r\_k} \in \mathbb{P}} \vert \lbrace j \geq 1 \mid g\_{j-1} \in (h, w) \And g\_j \not \in (h, w) \rbrace \vert.
$$

这里 $\mathbb{P}$ 表示时段 $i$ 的轨迹集合。$T\_r: g\_1 \rightarrow g\_2 \rightarrow \cdots \rightarrow g\_{\vert T\_r \vert}$ 是 $\mathbb{P}$ 中的一条轨迹，$g\_j$ 是坐标；
$g\_j \in (h, w)$ 表示点 $g\_j$ 在网格 $(h, w)$ 内，反之亦然；$\vert \cdot \vert$ 表示集合的基数。

**Crowd Flow Prediction**: 给定历史观测值 $\lbrace \mathbf{X}\_i \mid i=1,2,\cdots, n-1 \rbrace$，预测 $\mathbf{X}\_n$。

ST-ResNet 包含四个组件，*closeness*, *period*, *trend* 和 外部因素单元。每个组成部分通过一个分支的残差单元或全连接层预测出一个流量地图。然后模型使用一个线性组合作为末端融合方式融合这些预测值。ST-ResNet 的外部因素包含了天气、假期事件、元数据。

卷积神经网络的卷积核通常很小，意味着他们不能直接捕获远距离的空间依赖。然而，大范围的空间依赖在城市中很重要。另一方面，ST-ResNet 忽略了人口流动的在位置上的影响。此外，ST-ResNet 的末端融合机制导致了模型交互上的缺点以及参数的低效，还有模型的不稳定的问题。

# Our Model

图 2 展示了我们模型的框架。主要有三个部分：流量输入、SemanticPlus 和 ResPlus 单元。流量慎入包含 *closeness, period, terend*，由于数据的时间范围限制可以减少为 *closeness, period*。SemanticPlus 包含 POI 分布和时间信息。ResPlus 单元可以捕获远距离空间依赖。每个区域的流入和流出流量通过每小时或者每半小时统计得到流量地图的时间序列。这些流量地图通过 Min-Max 归一化处理到 $[-1, 1]$。如图 2 所示，人口分布地图通过近期时间、近邻历史、远期历史选择后作为输入放入模型。不同类型的 POI 分布通过 Min-Max 归一化到 $[0, 1]$。如图 2 做部分所示，POI 分布地图通过时间信息赋予了不同的权重。之后，POI 信息和人流信息通过早融合后放入堆叠的 ResPlus 单元中。最后，ResPlus 单元不同级别的特征融合后进入卷积部分，然后通过 Tanh 映射到 $[-1, 1]$。下面会介绍细节。

![Figure2](/images/deepstn-context-aware-spatial-temporal-neural-network-for-crowd-flow-prediction-in-metropolis/Fig2.JPG)

## ResPlus

很多处理人口流量预测的深度学习模型主要包含两个部分：基于 RNN 的结构，像 ConvLSTM 和 Periodic-CRN，以及基于 CNN 的结构，如 Deep-ST 和 ST-ResNet。但是，训练基于 RNN 结构的模型费时。因此我们选用基于 CNN 的结构 ST-ResNet 作为我们的基础模型。

在这篇论文中，我们设计 ConvPlus 来捕获城市内远距离的空间依赖。如图 3，ResPlus 单元使用一个 ConvPlus 和一个典型卷积。我们尝试了 Batch Normalization 和 Dropout，为了简介没有在图里面画出来。

![Figure3](/images/deepstn-context-aware-spatial-temporal-neural-network-for-crowd-flow-prediction-in-metropolis/Fig3.JPG)

典型卷积的每个通道对应一个卷积核。卷积核使用这些核来计算地图上的互相关系数，比如捕获梯度上的特征。卷积核的大小一般很小。在 ST-ResNet 和 DeepSTN+ 里面，卷积核的大小是 $3 \times 3$。但是城市中存在着远距离的依赖。人们可能坐地铁去上班。我们称这类关系叫远距离空间依赖关系。这种关系使得堆叠卷积难以有效地捕获这个关系。

如图 3 左部分所示，在 ConvPlus 结构中，我们将典型卷积的一些通道分离来捕获每个区域的远距离空间依赖。然后用一个全连接层直接捕获每两个区域之间的远距离空间依赖，在这层前面用一个池化层来减少参数。因此，在 ConvPlus 的输出有两类通道。ConvPlus 的输出有着和普通卷积一样的输出，可以用于下一个卷积的输入。

图 4 展示了两个不同区域的空间依赖热力图，分别是红色和黄色的星。这些目标区域不仅有区域上的依赖，还有一些和远处区域的远距离依赖。这也显示出不同的区域和地图上的其他区域有不一样的关系，这很难通过堆叠卷积有效地捕获。

![Figure4](/images/deepstn-context-aware-spatial-temporal-neural-network-for-crowd-flow-prediction-in-metropolis/Fig4.JPG)

因为 ConvPlus 有两类不同的输出通道，我们在 ResPlus 单元中使用 ConvPlus + Conv 而不是 ConvPlus + ConvPlus。没有 SemanticPlus 的 DeepSTN+ 形式化为：

$$
\widehat{\mathbf{X}} = f\_{Res}(f\_{EF}(\mathbf{X}^c + \mathbf{X}^p + \mathbf{X}^t)),
$$

三个 $\mathbf{X}$ 表示三种类型的历史地图——*closeness, period, trend*。$\widehat{\mathbf{X}}$ 表示预测出的流量地图。$+$ 表示拼接操作。$f\_{EF}$ 表示用来早融合不同类型信息的卷积函数，$f\_{Res}$ 表示一个堆叠的 ResPlus 单元。

## SemanticPlus

POI 在人口流动上有很强烈的影响，这些影响随时间变化而变化。因此，我们继承这个先验知识到模型内。我们手机了包括类型、数量、位置的 POI 信息。然后统计每个网格内 POI 的数量，使用一个一维向量表示每种 POI 的分布。图 5 展示了北京的流量分布地图和餐饮分布地图。它们的分布很相似，并且互相关系数有 0.87，暗示了它们之间的潜在关系。

![Figure5](/images/deepstn-context-aware-spatial-temporal-neural-network-for-crowd-flow-prediction-in-metropolis/Fig5.JPG)

我们使用一个时间向量来表示每个人口流量地图的时间。时间向量包含两个部分：一个 one-hot 向量表示一天中的各个时间，如果时段按小时走，那长度就是 24；另一个 one-hot 向量表示是一周中的哪天，长度是 7。一个时间向量拼接了这两个向量。

为了建模对流量地图有变化的时间影响的 POI 信息，我们将时间向量转换为 POI 的影响强度。我们使用大小为 $PN \times H \times W$ 的 $\mathbf{X}^s$ 来表示 POI 地图（$PN$ 表示 POI 的类数，$H$ 和 $W$ 是网格的行数和列数，一个向量 $\bf{I}$ 用来表示时间向量，大小为 $PN$ 的向量 $\bf{R}$ 表示 POI 的影响强度。因此，我们有带有时间权重的 POI 分布，形式化如下：

$$
\mathbf{S} = \mathbf{X}^s \ast \mathbf{R} = \mathbf{X}^s \ast f\_t(\mathbf{I})
$$

函数 $f\_t()$ 将时间向量转换为表示 POI 影响强度的向量。$\ast$ 表示每个 POI 分布地图会被附上一个权重，表示 POI 的影响强度。我们假设同一类在不同的区域的 POI 有相同的时间模式。因此，一个类别的 POI 分布地图会有相同的权重。图 6 展示了娱乐和居住区的影响强度。影响强度在一周内随时间的变化而变化，每天存在着一些典型的模式。很多人早上去上班，工作结束后回家，所以每天早上和下午住宅区有明显的两个峰。对比居住区，娱乐区的影响相对稳定。

![Figure6](/images/deepstn-context-aware-spatial-temporal-neural-network-for-crowd-flow-prediction-in-metropolis/Fig6.JPG)

## Fusion

三组件应该用更复杂的融合方式，而不是线性组合。这些带有 POI 信息的流量信息也有复杂的交互。为了建模这种相互影响，我们使用早融合而不是末端融合使得不同的信息能更早的融合起来。早融合减少了大约三分之二的参数。此外，ST-ResNet 有些时候不能收敛。我们发现这个问题可以通过早融合减少参数来简化模型解决。考虑到不同层的特征有不同的函数，我们在模型末端设定了一个多尺度的融合机制。这里我们形式化描述整个网络：

$$
\widehat{\mathbf{X}} = f\_{con}(f\_{Res}(f\_{EF}(\mathbf{X}^c + \mathbf{X}^p + \mathbf{X}^t + \mathbf{S}))),
$$

函数 $f\_{EF}$ 表示一个早融合使用的卷积操作，在早融合之前压缩了通道数。函数 $f\_{con}$ 表明了最后的多尺度融合，表示卷积层后的一个拼接层。$\bf{S}$ 表示 SemanticPlus 的输出，即 带有时间权重的 POI 分布。

## Training

算法 1 描述了训练过程。前 7 行是构建训练集和 POI 信息，模型通过 Adam 训练（8-12 行）

![Alg1](/images/deepstn-context-aware-spatial-temporal-neural-network-for-crowd-flow-prediction-in-metropolis/Alg1.JPG)

# Performance Evaluation

这部分，我们在两个数据集上不同城市的不同类型的流量上做了大量的实验，为了回答三个研究问题：
- 我们的提出的 DeepSTN+ 是否比现存的方法好？
- ResPlus, SemanticPlus, 早融合是怎么提升预测结果的？
- DeepSTN+ 的超参数如何影响预测结果？

## Datasets

表 1 包含了数据。每个数据有两个子集：流量轨迹和 POI 信息。

![Table1](/images/deepstn-context-aware-spatial-temporal-neural-network-for-crowd-flow-prediction-in-metropolis/Table1.JPG)

***MobileBJ:*** 数据是中国一个很流行的社交网络应用商提供的，时间范围是4 月 1 日到 4 月 30 日。记录了用户请求区域服务时的位置。我们用定义 2 转换成了网格流量。我们选择最后一周的数据作为测试集，前面的作为训练集。表 2 展示了这个数据集的 17 类 POI 信息。

***BikeNYC:*** NYC 的自行车数据，2014 年，4 月 1 日到 9 月 30 日。数据包含了旅途时长，出发和到达站的 ID，起始和结束时间。最后 14 天的数据用来测试，其他的训练。我们选了 9 类 POI 信息。

![Table2](/images/deepstn-context-aware-spatial-temporal-neural-network-for-crowd-flow-prediction-in-metropolis/Table2.JPG)

## Baselines

- HA
- VAR
- ARIMA
- ConvLSTM
- ST-ResNet
  
## Metrics and Parameters

- RMSE

$$
RMSE = \sqrt{\frac{1}{T} \sum^T\_{i=1} \Vert \mathbf{X}\_i - \widehat{X}\_i \Vert^2\_2},
$$

- MAE

$$
MAE = \frac{1}{T} \sum^T\_{i=1} \vert \mathbf{X}\_i - \widehat{\mathbf{X}}\_i \vert,
$$

RMSE 作为 loss function。

![Table3](/images/deepstn-context-aware-spatial-temporal-neural-network-for-crowd-flow-prediction-in-metropolis/Table3.JPG)

表 3 展示了不同的参数设置。

![Table4](/images/deepstn-context-aware-spatial-temporal-neural-network-for-crowd-flow-prediction-in-metropolis/Table4.JPG)

![Table5](/images/deepstn-context-aware-spatial-temporal-neural-network-for-crowd-flow-prediction-in-metropolis/Table5.JPG)

![Figure7](/images/deepstn-context-aware-spatial-temporal-neural-network-for-crowd-flow-prediction-in-metropolis/Fig7.JPG)