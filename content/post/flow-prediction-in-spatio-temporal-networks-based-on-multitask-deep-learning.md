---
categories:
- 论文阅读笔记
date: 2019-04-19 16:40:41+0000
draft: false
math: true
tags:
- deep learning
- ResNet
- Spatial-temporal
- graph convolutional network
- Graph
- Time Series
title: Flow Prediction in Spatio-Temporal Networks Based on Multitask Deep Learning
---
TKDE 2019，网格流量预测，用一个模型同时预测每个网格的流入/流出流量和网格之间的转移流量，分别称为顶点流量和边流量，同时预测这两类流量是本文所解决的多任务预测问题。本文提出的是个框架，所以里面用什么组件应该都是可以的，文章中使用了 FCN。使用两个子模型分别处理顶点流量和边流量预测问题，使用两个子模型的输出作为隐藏状态表示，通过拼接或加和的方式融合，融合后的新表示再分别输出顶点流量和边流量。这篇文章和之前郑宇的文章一样，考虑了三种时序性质、融合了外部因素。损失函数从顶点流量预测值和真值之间的差、边流量预测值和真值之间的差、顶点流量预测值之和与边流量的预测值之差三个方面考虑。数据集是北京和纽约的出租车数据集。 [Flow Prediction in Spatio-Temporal Networks Based on Multitask Deep Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8606218)

<!--more-->

**Abstract**——预测流量（如车流、人流、自行车流）包括结点的流入、流出流量以及不同的结点间的转移，在交通运输系统中的时空网络里扮演着重要的角色。然而，这个问题受多方面复杂因素影响，比如不同地点的空间关系、不同时段的时间关系、还有像活动和天气这样的外部因素，所以这是个有挑战性的问题。此外，一个结点的流量（结点流量）和结点之间的转移（边流量）互相影响。为了解决这个问题，我们提出了一个多任务的深度学习框架可以同时预测一个时空网络上的结点流量和边流量。基于全卷积网络，我们的方法设计了两个复杂的模型分别处理结点流量预测和边流量预测。这两个模型通过组合中间层的隐藏表示连接，而且共同训练。外部因素通过一个门控融合机制引入模型。在边流量预测模型上，我们使用了一个嵌入组件来处理顶点间的系数转移问题。我们在北京和纽约的出租车数据集上做了实验。实验结果显示比11种方法都好。

# 1 Introduction

时空网络（ST-networks），如运输网络和传感器网络，在世界上到处都是，每个点有个空间坐标，每个边具有动态属性。时空网络中的流量有两种表示，如图 1，顶点流量（一个结点的流入和流出流量）和边流量（结点间的转移流量）。在运输系统中，这两类流量可通过4种方式测量，1. 近邻道路的车辆数，2. 公交车的旅客数，3. 行人数，4. 以上三点。图1b 是一个示意图。取顶点 $r\_1$ 为例，我们可以根据手机信令和车辆 GPS 轨迹分别计算得到流入流量是 3，流出流量是 3。$r\_3$ 到 $r\_1$ 的转移是 3，$r\_1$ 到 $r\_2$ 和 $r\_4$ 的转移是 2 和 1。因此，如图1c所示，我们能拿到两种类型的流量，四个结点的流入和流出分别是 $(3,3,0,5)$ 和 $(3,2,5,1)$。所有的边转移都看作是在有向图上发生的。

![Figure1](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Fig1.JPG)

预测这类的流量对公共安全，交通管理，网络优化很重要。取人口流动做一个例子，2015 年跨年夜的上海，踩踏事故导致 36 人死亡。如果能预测每个区域之间的人流转移，这样的悲剧就可以通过应急预案避免或减轻。

然而，同时预测所有结点和边的转移是很难的：

1. **Scale and complexity**: 一个地方的流入和流出依赖于它的邻居，有近邻的也有遥远的，因为人们会在这些区域之间转移，尤其是有活动的时候。给定一个城市，有 $N$ 个地点，$N$ 很大，那么就有 $N^2$ 种转移方式，尽管这些转移可能不会同时发生。因此，预测地点的流量，要么是流入、流出或是转移流量，我们需要考虑地点之间的依赖关系。而且，预测也考虑过去时段的流量。此外，我们不能单独地预测每个地点的流量，因为城市内的地点间是相连的，相关的，互相影响的。复杂度和尺度都是传统机器学习模型，如概率图模型在解决这个问题时面临的巨大挑战。
2. **Model multiple correlations and external factors**: 我们需要对三种关系建模来处理预测问题。第一个是不同地点流量的空间相关性，包含近邻和遥远的。第二个是一个地点不同时段的流量间的时间关系，包括时间近邻、周期和趋势性。第三，流入流出流量和转移流量高度相关，互相影响。一个区域的转入流量之和是这个区域的流入流量。精确地预测一个区域的流出流量可以让预测其他区域的转移流量更精确，反之亦然。此外，这些流量受外部因素影响，如活动、天气、事故等。如何整合这些信息还是个难题。
3. **Dynamics and sparsity**: 由于 $N^2$ 种情况，区域间随时间改变的转移流量比流入流出流量要大得多。一个地点和其他地点间的转移会在接下来的时段发生，可能是 $N^2$ 中的很小一部分（稀疏）。预测这样的稀疏转移也是个难题。

为了解决上述挑战，我们提出了多任务深度学习框架MDL（图4）来同时预测顶点流量和边流量。我们的贡献有三点：
- MDL 设计了一个深度神经网络来预测顶点流量（命名为 NODENET），另一个深度神经网络预测边流量（命名为 EDGENET）。通过将他们的隐藏状态拼接来连接这两个深度神经网络，并一同训练。此外，这两类流量的相关性通过损失函数中的正则项来建模。基于深度学习的模型可以处理复杂性和尺度等问题，同时多任务框架增强了每类流量的预测性能。
- NODENET 和 EDGENET 都是 three-stream 全卷积网络（3S-FCNs），closeness-stream, period-stream, trend-stream 捕获三种不同的时间相关性。每个 FCN 也同时捕获近邻和遥远的空间关系。一个门控组件用来融合时空相关性和外部因素。为了解决转移稀疏的问题，EDGENET 中我们设计了一个嵌入组件，用一个隐藏低维表示编码了稀疏高维的输入。
- 我们在北京和纽约的 taxicab data 上评估了方法。结果显示我们的 MDL 超越了其他 11 种方法。

表 1 列出了这篇文章中出现的数学符号。

![Table1](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Table1.JPG)

# 2 Problem Formulation

***Definition 1(Node).*** 一个空间地图基于经纬度被分成 $I \times J$ 个网格，表示为 $V = \lbrace r\_1, r\_2, ..., r\_{I\times J} \rbrace$，每个元素表示一个空间节点，如图2(a)。

![Figure2](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Fig2.JPG)

令 $(\tau, x, y)$ 为时空坐标，$\tau$ 表示时间戳，$(x, y)$ 表示空间点。一个物体的移动可以记为一个按时间顺序的空间轨迹，起点和终点表示为 $s = (\tau\_s, x\_s, y\_s)$ 和 $e = (\tau\_e, x\_e, y\_e)$，表示出发地和目的地。$\mathbb{P}$ 表示所有的起止对。

***Definition 2(In/out flows).*** 给定一组起止对 $\mathbb{P}$。$\mathcal{T} = \lbrace t\_1, \dots t\_T\rbrace$ 表示一个时段序列。对于地图上第 $i$ 行第 $j$ 列的顶点 $r\_{ij}$，时段 $t$ 流出和流入的流量分别定义为：

$$\tag{1}
\mathcal{X}\_t(0, i, j) = \vert \lbrace (s,e) \in \mathbb{P} : (x\_s, y\_s) \in r\_{ij} \wedge \tau\_s \in t \rbrace \vert
$$

$$\tag{2}
\mathcal{X}\_t(1, i, j) = \vert \lbrace (s,e) \in \mathbb{P} : (x\_e, y\_e) \in r\_{ij} \wedge \tau\_e \in t \rbrace \vert
$$

其中 $\mathcal{X}\_t(0, :, :)$ 和 $\mathcal{X}\_t(1, :, :)$ 表示流出和流入矩阵。$(x, y) \in r\_{ij}$ 表示点 $(x, y)$ 在顶点 $r\_{ij}$ 上，$\tau\_e \in t$ 表示时间戳 $\tau\_e$ 在时段 $t$ 内。流入和流出矩阵在特定时间的矩阵如图2。

考虑两类流量（流入和流出），一个随时间变化的空间地图一般表示一个时间有序的张量序列，每个张量对应地图在特定时间的一个快照。详细来说，每个张量包含两个矩阵：流入矩阵和流出矩阵，如图 2 所示。

让 $V$ 表示时空网络中的顶点集，$N \triangleq \vert V \vert = I \times J$ 是顶点数。一个时间图包含 $T$ 个离散的不重叠的时段，表示为有向图 $G\_{t\_1}, \dots G\_{t\_T}$ 的时间有序序列。图 $G\_t = (V, E\_t)$ 捕获了时段 $t$ 时空系统上的拓扑状态。对于每个图 $G\_t$ (其中 $t = t\_1, \dots, t\_T$) 存在一个对应的权重矩阵 $\mathbf{S}\_t \in \mathbb{R}^{N \times N}$，表示时段 $t$ 的带权有向边。在我们的研究中，时段 $t$ 顶点 $r\_s$ 到顶点 $r\_e$ 的边的权重，是一个非负标量，表示 $r\_s$ 到 $r\_e$ 的 *transition*，时段 $t$ 上两个顶点间没有连接的话，对应的元素在 $\mathbf{S}\_t$ 中为 0。

***Definition 3 (Transition).*** 给定一组起止点对 $\mathbb{P}$。$\mathcal{T} = \lbrace t\_1, \dots, t\_T \rbrace$ 是一组时段的序列。$\mathbf{S}\_t$ 是时段 $t$ 的转移矩阵，$r\_s$ 到 $r\_e$ 之间的转移表示为 $\mathbf{S}\_t(r\_s, r\_e)$，定义为：

$$\tag{3}
\mathbf{S}\_t(r\_s, r\_e) = \vert \lbrace (s,e) \in \mathbb{P} : (x\_s, y\_s) \in r\_s \wedge (x\_e, y\_e) \in r\_e \wedge \tau\_s \in t \wedge \tau\_e \in t \rbrace \vert
$$

其中 $r\_s, r\_e \in V$ 是起始顶点和终止顶点。$(x, y) \in r$ 表示点 $(x, y)$ 在网格 $r$ 上。$\tau\_s \in t$ 和 $\tau\_e \in t$ 表示时间戳 $\tau\_s$ 和 $\tau\_e$ 都在时段 $t$ 内。我们考虑转移至发生在一个特定的时段内。因此，对于实际应用来说，我们可以预测起始和结束都发生在未来的转移。

## 2.1 Converting time-varying graphs into tensors

我们将每个时间上的图转为张量。给定时间 $t$ 有向图 $G\_t = (V, E\_t)$，我们先做展开，然后计算有向带权矩阵（转移矩阵 $\mathbf{S}\_t$），最后给定一个张量 $\mathcal{M}\_t \in \mathbf{R}^{2N \times I \times J}$。图 3 是示意图。(a)给定时间 $t$ 4 个顶点 6 条边的图。(b)首先展开成有向图。(c)对每个顶点，有一个流入的转移，还有个流出的转移，由一个向量表示（维度是8）。取 $r\_1$ 为例，它的流出和流入转移向量分别为 $[0, 2, 0, 1]$ 和 $[0, 0, 3, 0]$，拼接后得到一个向量 $[0, 2, 0, 1, 0, 0, 3, 0]$，包含流出和流入的信息。(d)最后，我们将矩阵 reshape 成一个张量，每个顶点根据原来地图有一个固定的空间位置，保护了空间相关性。

![Figure3](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Fig3.JPG)

## 2.2 FLow Prediction Problem

流量预测，简单来说，是时间序列问题，目标是给定历史 $T$ 个时段的观测值，预测 $T+1$ 时段每个区域的流量。但是我们的文章中流量有两个层次，流入和流出以及区域间的转移流量。我们的目标是同时预测这些流量。此外，我们还融入了外部因素如房价信息，天气状况，温度等等。这些外部因素可以收集并提供一些额外有用的信息。相关的符号在表 1 之中。

***Problem 1.*** 给定历史观测值 $\lbrace \mathcal{X}\_t, \mathcal{M}\_t \mid t = t\_1, \dots, t\_T \rbrace$，外部特征 $\mathcal{E}\_T$，我们提出一个模型共同预测 $\mathcal{X}\_{t\_{T+1}}$ 和 $\mathcal{M}\_{t\_{T+1}}$。

# 3 Multitask Deep Learning

图 4 展示了我们的 MDL 框架，包含 3 个组成部分，分别用于数据转换，顶点流量建模，边流量建模。我们首先将轨迹（或订单）数据转换成两类流量，i) 顶点流量表示成有时间顺序的张量序列 $\lbrace\mathcal{X}\_t \mid t = t\_1, \dots, t\_T \rbrace$ (1a); ii) 边流量是一个有时间顺序的图序列（转移矩阵）$\lbrace\mathbf{S}\_t \mid t = t\_1, \dots, t\_T \rbrace$ (2a)，之后再根据 2.1 节的方法转换为张量的序列 $\lbrace\mathcal{M}\_t \mid t = t\_1, \dots, t\_T \rbrace$ (2b)。这两类像视频一样的数据之后放到 NODENET 和 EDGENET 中。以 NODENET 为例，它选了三个不同类型的片段，放入 3S-FCN 中，对时间相关性建模。在这个模型中，每部分的 FCN 可以通过多重卷积捕获空间相关性。NODENET 和 EDGENET 中间的隐藏表示通过一个 BRIDGE 组件连接，使两个模型可以共同训练。我们使用一个嵌入层来处理转移稀疏的问题。一个门控融合组件用来整合外部信息。顶点流量和边流量用一个正则化来建模。

![Figure4](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Fig4.JPG)

## 3.1 EDGENET

根据上述的转换方法，每个时段的转移图可以转换成一个张量 $\mathcal{M}\_t \in \mathbb{R}^{2N \times I \times J}$。对于每个顶点 $r\_{ij}$，它最多有 $2N$ 个转移概率，包含 $N$ 个流入和 $N$ 个流出。然而，对于一个确定的时段，顶点间的转移是稀疏的。受 NLP 的嵌入方法启发，我们提出了使用空间嵌入方法，解决这样的稀疏和高维问题。详细来说，空间嵌入倾向于学习一个将 $2N$ 维映射到 $k$ 维的函数：

$$\tag{4}
\mathcal{Z}\_t(:, i, j) = \mathbf{W}\_m \mathcal{M}\_t (:, i, j) + \mathbf{b}\_m, 1 \leq i \leq I, 1 \leq j \leq J
$$

其中 $\mathbf{W}\_m \in \mathbb{R}^{k \times 2N}$ 和 $\mathbf{b}\_m \in \mathbb{R}^k$ 是参数。所有的结点共享参数。$\mathcal{M}\_t(:, i, j) \in \mathbb{R}^{2N}$ 表示 $(i, j)$ 的向量。

流量，比如城市中的交通流，总是受时空依赖关系影响。为了捕获不同的时间依赖（近邻、周期、趋势），Zhang et al. 提出了深度时空残差网络，沿时间轴选择不同的关键帧。受这点的启发，我们选择近邻、较近、远期关键帧来预测时段 $t$，分别表示为 $M^{dep}\_t = \lbrace M^{close}\_t, M^{period}\_t, M^{trend}\_t \rbrace$，如下：

- **Closeness** dependents:
  $$M^{close}\_t = \lbrace \mathcal{Z}\_{t-l\_c}, \dots, \mathcal{Z}\_{t-1} \rbrace$$
- **Period** dependents:
  $$M^{period}\_t = \lbrace \mathcal{Z}\_{t-l\_p}, \mathcal{Z}\_{t-(l\_p - 1) \cdot p}, \dots, \mathcal{Z}\_{t-p} \rbrace$$
- **Trend** dependents:
  $$M^{trend}\_t = \lbrace \mathcal{Z}\_{t-l\_q \cdot q}, \mathcal{Z}\_{t-(l\_q - 1)\cdot q}, \dots, \mathcal{Z}\_{t-q} \rbrace$$

其中 $p$ 和 $q$ 是周期和趋势范围。$l\_c$, $l\_p$ 和 $l\_q$ 是三个序列的长度。

输出（即下个时段的预测）和输入有相同的分辨率。这样的人物和图像分割问题很像，可以通过全卷积网络 (FCN) [22] 处理。

受到这个启发，我们提出了三组件的 FCN，如图 4，来捕获时间近邻、周期和趋势依赖。每个组件都是个 FCN，包含了很多卷积（图 5）。根据卷积的性质，一个卷积层可以捕获空间近邻关系。随着卷积层数的增加，FCN 可以捕获更远的依赖，甚至是城市范围大小的空间依赖。然而，这样的深层卷积网络很难训练。因此我们使用残差连接来帮助训练。类似残差网络中的残差连接，我们使用一个包含 BN，ReLU，卷积的块。令三个近邻、周期、趋势三组件的输出分别为 $\mathcal{M}\_c$, $\mathcal{M}\_p$, $\mathcal{M}\_q$。不同的顶点在近邻、周期、趋势上可能有不同的性质。为了解决这个问题，我们提出使用一个基于参数矩阵的融合方式（图 4 中的 PM 融合）：

$$\tag{5}
\mathcal{M}\_{fcn} = \mathbf{W}\_c \odot \mathcal{M}\_c + \mathbf{W}\_p \odot \mathcal{M}\_p + \mathbf{W}\_q \odot \mathcal{M}\_q
$$

其中 $\odot$ 是哈达玛积，$\mathbf{W}$ 是参数，调整三种时间依赖关系的影响。

![Figure5](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Fig5.JPG)

## 3.2 NODENET and BRIDGE

类似 EDGENET，NODENET 也是一个 3S-GCN，我们选择近邻、较近、遥远的关键帧作为近邻、周期、趋势依赖。区别是 NODENET 没有嵌入层因为输入的通道数只有 2。这三种不同的依赖放入三个不同的 FCN 中，输出通过 PM 融合组件融合（图 4）。然后，得到 3S-FCN 的输出，表示为 $\mathcal{X}\_{fcn} \in \mathbb{R}^{C\_x \times I \times J}$。

考虑顶点流量与边流量的相关性，所以从 NODENET 和 EDGENET 学习到的表示应该被连起来。为了连接 NODENET 和 EDGENET，假设 NODENET 和 EDGENET 的隐藏表示分别为 $\mathcal{X}\_{fcn}$ 和 $\mathcal{M}\_{fcn}$。我们提出两种融合方法：

**SUM Fusion:** 加和融合方法直接将两种表示相加：

$$\tag{6}
\mathcal{H}(c, :, :) = \mathcal{X}\_{fcn}(c, :, :) + \mathcal{M}\_{fcn}(c, :, :), c = 0, \dots, C - 1
$$

其中 $C$ 是 $\mathcal{X}\_{fcn}$ 和 $\mathcal{M}\_{fcn}$ 的通道数，$\mathcal{H} \in \mathbb{R}^{C \times I \times J}$。显然这种融合方法受限于两种表示必须有相同的维度。

**CONCAT Fusion:** 为了从上述的限制中解脱，我们提出了另一种融合方法。顺着通道拼接两个隐藏表示：

$$\tag{7}
\mathcal{H}(c, :, :) = \mathcal{X}\_{fcn}(c, :, :), c=0, \dots, C\_x - 1
$$

$$\tag{8}
\mathcal{H}(C\_x + c, :, :) = \mathcal{M}\_{fcn}(c, :, :), c=0, \dots, C\_m - 1
$$

$C\_x$ 和 $C\_m$ 分别是两个隐藏表示的通道数。$\mathcal{H} \in \mathbb{R}^{(C\_x + C\_m) \times I \times J}$。拼接融合实际上可以通过互相强化更好地融合顶点流量和边流量。像 BRIDGE 一样我们也讨论了其他的融合方式（4.3 节）。

在拼接融合中，我们在 NODENET 和 EDGENET 中分别加了一层卷积。卷积用来将合并的隐藏特征 $\mathcal{H}$ 映射到 不同通道大小的输出上，即 $\mathcal{X}\_{res} \in \mathbb{R}^{2 \times I \times J}$ 和 $\mathcal{M}\_{res} \in \mathbb{R}^{2N \times I \times J}$，如图 6。

![Figure6](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Fig6.JPG)

## 3.3 Fusing External Factors Using a Gating Mechanism

外部因素，活动、天气会影响时空网络不同区域的流量。举个例子，一起事故可能会阻塞一个局部区域的交通，一场暴风雨可能会减少整个城市的流量。这样的外部因素就像一个开关，如果它打开了那流量会产生巨大的变化。基于这个思路，我们开发了一种基于门控机制的融合，如图 6 所示。时间 $t$ 的外部因素表示为 $\mathcal{E}\_t \in \mathbb{R}^{l\_e \times I \times J}$，$\mathcal{E}\_t(:, i, j) \in \mathbb{R}^{l\_e}$ 表示一个特定顶点的外部信息。我们可以通过下式获得 EDGENET 的门控值：

$$\tag{9}
\mathbf{F}\_m(i, j) = \sigma(\mathbf{W}\_e(:, i, j) \cdot \mathcal{E}\_t(:, i, j) + \mathbf{b}\_e(i, j)), 1 \leq i \leq I, 1 \leq j \leq J
$$

其中 $\mathbf{W}\_e \in \mathbb{R}^{l\_e \times I \times J}$ 和 $\mathbf{b}\_e \in \mathbb{R}^{I \times J}$ 是参数。$\mathbf{F}\_m \in \mathbb{R}^{I \times J}$ 是 GATING 的输出，$\mathbf{F}\_m(i, j)$ 是对应时空网络中结点 $r\_{ij}$ 的门控值。$\sigma(\cdot)$ 是 sigmoid 激活函数，$\cdot$ 是两向量的内积。

然后我们使用 PRODUCT 融合方式：

$$\tag{10}
\hat{\mathcal{M}}\_t(c, :, :) = \text{tanh}(\mathbf{F}\_m \odot \mathcal{M}\_{Res}(c, :, :)), c = 0, \dots, 2N - 1
$$

类似的，NODENET 最后在时间 $t$ 的预测结果为：

$$\tag{11}
\hat{\mathcal{X}}\_t(c, :, :) = \text{tanh} (\mathbf{F}\_x \odot \mathcal{X}\_{Res}(c, :, :)), c = 0, 1
$$

其中 $\mathbf{F}\_x \in \mathbb{R}^{I \times J}$ 是 GATING 的另一个输出。对于顶点流量和边流量使用不同的门控值的一个原因是外部因素对流入/流出流量和不同地点之间的转移流量的影响是不一致的。

## 3.4 Losses

令 $\phi$ 为 EDGENET 中所有的参数，我们的目标是通过最小化目标函数学习这些参数：

$$\tag{12}
\mathop{\mathrm{argmin}}\limits\_{\phi} \mathcal{J}\_{edge} = \sum\_{t \in \mathcal{T}}\sum^{2N-1}\_{c=0} \Vert Q^c\_t \odot (\hat{\mathcal{M}}\_t(c, :, :) - \mathcal{M}\_t(c, :, :)) \Vert^2\_F
$$

其中 $Q^c\_t$ 是指示矩阵，表示 $\mathcal{M}\_t(c, :, :)$ 中所有非零元素。$\mathcal{T}$ 是可用的时段，$\Vert \cdot \Vert\_F$ 是矩阵的 F 范数。

类似的，$\theta$ 是 NODENET 的参数，目标函数是：

$$\tag{13}
\mathop{\mathrm{argmin}}\limits\_{\theta} \mathcal{J}\_{node} = \sum\_{t \in \mathcal{T}}\sum^1\_{c=0} \Vert P^c\_t \odot (\hat{\mathcal{X}}\_t(c, :, :) - \mathcal{X}\_t(c, :, :)) \Vert^2\_F
$$

其中 $P^c\_t$ 是指示矩阵，表示 $\mathcal{X}\_t(c, :, :)$ 中所有非零元素。我们知道对于一个结点来说，它的转入流量之和就是它的流入流量，转出流量之和就是流出流量。定义 2 中定义，$\hat{\mathcal{X}}\_t(0, :, :)$ 和 $\hat{\mathcal{X}}\_t(1, :, :)$ 分别是流出和流入矩阵。根据 2.1 节定义的方法构建转移矩阵，可知前 $N$ 个通道表示转出流量，后 $N$ 个通道表示转入流量。因此，有下面的损失函数：

$$\tag{14}
\mathop{\mathrm{argmin}}\limits\_{\theta, \phi} \sum\_{t \in \mathcal{T}} \sum\_i \sum\_j (\Vert \hat{\mathcal{X}}\_t(0, i, j) - \sum^{N-1}\_{c=0} \hat{\mathcal{M}}\_t(c,i,j) \Vert^2 + \Vert \hat{\mathcal{M}}\_t(1,i,j) - \sum^{2N-1}\_{c=N} \hat{\mathcal{M}}\_t(c,i,j) \Vert^2)
$$

或者等价的可以写成

![EQ15](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/EQ1.JPG)

最后，我们获得融合的损失：

$$\tag{16}
\mathop{\mathrm{argmin}}\limits\_{\theta, \phi} \lambda\_{node} \mathcal{J}\_{node} + \lambda\_{edge} \mathcal{J}\_{edge} + \lambda\_{mdl} \mathcal{J}\_{mdl}
$$

其中，$\lambda\_{node}$, $\lambda\_{edge}$, $\lambda\_{mdl}$ 是可调节的参数。

### 3.4.1 Optimization Algorithm

![Alg1](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Alg1.JPG)

算法 1 是 MDL 的训练过程。1-4 行是构建训练样例。7-8 行是用批量样本优化目标函数。

# 4 Experiments

两个数据集 **TaxiBJ** 和 **TaxiNYC**，看表 2。我们使用 RMSE 和 MAE 作为评价指标。

![Table2](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Table2.JPG)

## 4.1 Settings

### 4.1.1 Datasets

我们使用表 3 中的两个数据集。每个数据集包含两个子集，轨迹/出行和外部因素，细节如下：

- **TaxiBJ**: 北京出租车 GPS 轨迹数据有四个时段：20130101-20131030, 20140301-20140630, 20150501-20150630, 201501101-20160410。我们用最后 4 个星期作为测试集，之前的数据作为训练集。
- **TaxiNYC**: NYC 2011 到 2014 年的出租车订单数据。订单数据包含上车和下车的时间。上车和下车地点。最后四个星期作为测试集。

![Table3](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Table3.JPG)

### 4.1.2 Baselines

HA, ARIMA, SARIMA, VAR, RNN, LSTM. GRU, ST-ANN, ConvLSTM, ST-ResNet, MRF.

### 4.1.3 Preprocessing

MDL 的输出，我们用 $\text{tanh}$ 作为最后的激活函数。我们用最大最小归一化。评估的时候，将预测值转换为原来的值。对于外部因素，使用 one-hot，假期和天气放入二值向量中，用最大最小归一化把温度和风速归一化。

### 4.1.4 Hyperparameters

$\lambda\_{node} = 1$ 和 $\lambda\_{edge} = 1$，$\lambda\_{mdl} = 0.0005$，$p$ 和 $q$ 按经验设定为一天和一周。三个依赖序列的长度分别为 $l\_c \in \lbrace 1, 2, 3\rbrace$, $l\_p \in \lbrace 1,2,3 \rbrace$, $l\_q \in \lbrace 1,2,3 \rbrace$。卷积的数量是 5 个。训练集的 90% 用来训练，10% 来验证，用早停选最好的参数。然后使用所有的数据训练模型。网络参数通过随机初始化，Adam 优化。batch size 32。学习率 $\lbrace 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005 \rbrace$。

### 4.1.5 Evaluation Metrics

RMSE 和 MAE。

## 4.2 Results

![Table4](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Table4.JPG)

**Node Flow Prediction.** 我们先比流入和流出流量的预测。表 4 展示了两个数据集上的评价指标结果。MDL 和 MRF 比其他所有的方法多要好。我们的 MDL 在 NYC 的数据集上明显比 MRF 好。BJ 的数据集上，MDL 比 MRF 差不多。原因是 NYC 数据集比 BJ 数据集大了三倍。换句话说，在大的数据集上，我们的方法比 MRF 更好。我们也注意到训练 MRF 很好使，在 BJ 数据集上训练了一个星期。

![Table5](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Table5.JPG)

**Results of Edge Flow Prediction.** 表6 展示了边流量预测。边流量预测的实验很费时。MDL 比其他的都好。

![Table6](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Table6.JPG)

## 4.3 Evaluation on Fusing Mechanisms

融合 NODENET 和 EDGENET 有 CONCAT 和 SUM 两种方法。融合外部因素有 GATED 和 SIMPLE 融合，或者不使用。因此总共有 6 种方法。如表 7。使用同样的超参数设定。我们发现 CONCAT + GATING 比其他的方法好。

## 4.4 Evaluation on Model Hyper-parameters

### 4.4.1 Effect of Training Data Size

我们选了 NYC 3 个月，6 个月，1 年，3 年数据。$l\_c = 3$, $l\_p = 1$, $l\_q = 1$。图 8 是结果。我们观察到数据越多，效果越好。

![Figure8](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Fig8.JPG)

### 4.4.2 Effect of Network Depth

图 9 展示了网络深度在 NYC 3 个月数据集上的影响。网络越深，RMSE 会下降，因为网络越深越能捕获更大范围的空间依赖。然而，网络更深 RMSE 就会上升，这是因为网络加深后训练会变得困难。

![Figure9](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Fig9.JPG)

### 4.4.3 Effect of multi-task component

表 8 和图 10 展示了多任务组件的影响。

我们可以看到转移流量预测任务大多数情况下可以提升，$\lambda\_{node} = \lambda\_{edge} = 1$，$\lambda\_{mdl}=0.1$，我们的模型获得最好的效果，两种任务都获得更好的结果，证明了多任务可以互相提升。

![Table8](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Table8.JPG)

![Figure10](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Fig10.JPG)

## 4.5 Flow Predictions

图 11 描绘了我们的 MDL 在 NYC 上预测两个节点未来一小时的数据。结点 (10, 1)，总是比 (8, 3) 高。我们的模型在预测曲线上更精确。

![Figure11](/images/flow-prediction-in-spatio-temporal-networks-based-on-multitask-deep-learning/Fig11.JPG)