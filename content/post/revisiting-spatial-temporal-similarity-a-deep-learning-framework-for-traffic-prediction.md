---
categories:
- 论文阅读笔记
date: 2019-03-21 10:43:34+0000
draft: false
math: true
tags:
- deep learning
- Spatial-temporal
- graph convolutional network
- Graph
- Time Series
title: 'Revisiting Spatial-Temporal Similarity: A Deep Learning Framework for Traffic
  Prediction'
---
AAAI 2019。网格流量预测，两个问题：空间依赖动态性，另一个是周期平移。原文链接：[Revisiting Spatial-Temporal Similarity: A Deep Learning Framework for Traffic Prediction](http://export.arxiv.org/abs/1803.01254)
<!--more-->

# Abstract

由于大规模的交通数据越来越多，而且交通预测在实际中很重要，交通预测在 AI 领域引起越来越多的关注。举个例子，一个精确的出租车需求预测可以协助出租车公司预分配出租车。交通预测的关键在于如何对复杂的空间依赖和时间动态性建模。尽管两个因素在建模的时候都会考虑，当前的方法仍会做很强的假设，即空间依赖在时间上是平稳的，时间依赖是严格周期的。然而，实际中的空间依赖可能是动态的（即随时间的变化而变化），而且时间动态性可能从一个时段到另一个时段有波动。在这篇文章中，我们有两个重要发现：（1）区域间的空间依赖是动态的；（2）时间依赖虽说有天和周的模式，但因为有动态时间平移，它不是严格周期的。为了解决这两个问题，我们提出了一个新的时空动态网络（STDN），我们用一个门控机制学习区域间的动态相似性，用一个周期性平移的注意力机制来处理长期周期时间平移现象。据我们所知，这是第一个在一个统一的框架中解决这两个问题的工作。我们的实验结果证明了提出的方法是有效的。

# Introduction

交通预测是一个时空预测问题。精确的预测模型对很多应用都很重要。在传统交通预测问题中，给定历史数据（比如一个区域的流量，或一个交叉卡前几个月每小时的流量），预测未来一段时间的数据。这方面的研究已经有几十年了。在时间序列社区，ARIMA 和 Kalman filtering被广泛地应用到这一领域。然而这些早期方法是针对每个区域分别预测，最近的方法考虑了空间信息（比如针对近邻区域增加正则项）和外部因素（如地点信息，天气状况，地区活动）。然而，这些方法仍基于机器学习中的传统时间序列模型，不能很好的捕获复杂的非线性时空依赖）。

最近，深度学习方法在很多任务上取得了成功。比如，一些研究将城市交通看作是热力图的图片，使用 CNN 对非线性空间关系建模。为了对非线性时间关系建模，人们提出了基于 RNN 的框架。Yao et al。 更是提出了用 CNN 和 LSTM 同时处理时间和空间依赖的框架。

尽管考虑了同时对时空建模，现存的方法主要有两点不足。首先，区域间的空间依赖依赖于历史数据的相似性，模型学习到了一个静态的空间依赖。然而，区域间的依赖随时间是改变的。举个例子，早上，居民区和商业区之间的依赖关系强；深夜，关系就弱了。然而，这样的动态依赖在之前的研究中没有考虑。

另一个限制是现存的研究忽略了长期周期依赖。交通数据又很强的日和周周期性，基于这种周期性的依赖关系可能用于预测。然而，一个挑战是交通数据不是严格周期的。举个例子，周末的高峰通常发生在下午的后半段，不同的日子时间不一致，从4:30pm到6:00pm变化。尽管之前的研究考虑了周期性，他们没能考虑序列性的依赖和周期性中的时间平移。

为了解决前面提出的问题，我们提出了新的深度学习框架，时空动态网络用于交通预测。STDN 是基于时空神经网络的，使用局部 CNN 和 LSTM 分别处理时空信息。一个门控局部 CNN 使用区域间的动态相似性对空间依赖建模。一个周期平移的注意力机制用来学习长期周期依赖。通过注意力机制对长期周期信息和时间平移建模。我们的方法还用 LSTM 以层次的方式处理序列依赖。

我们再大型的真实数据集上做了评测，纽约出租车数据和纽约的共享单车数据。和 state-of-the-art 的全面对比展示了我们模型的性能。我们的贡献如下：
- 我们提出了一个流式门控机制对动态空间相似性建模。门控制信息在邻近区域传播。
- 我们提出了一个周期平移注意力机制，通过同时时可用长期周期信息和时间平移。
- 我们在几个真实数据集上开展了实验，效果好。

# Notations and Problem Formulation

我们将整个城市分为 $a \times b$ 个网格，一共 $n$ 个区域（$n = a \times b$），使用 $\lbrace 1,2,\dots,n \rbrace$ 表示他们。我们将整个时间周期分为 $m$ 个登场的连续时段。任何一个个体的移动，其本质上是整个城市交通的一部分，总是从一个区域出发，过一段时间到达目的地。我们在一个时段内给每个区域定义一个开始/结束流量作为区域出发/到达的移动发生次数。$y^s\_{i,t}$ 和 $y^e\_{i,t}$ 表示开区域 $i$ 在时段 $t$ 的开始/结束流量。此外，对个体旅行的聚合形成交通流，描述了区域对之间的考虑时间的移动。时段 $t$ 从区域 $i$ 开始的交通流在时段 $\tau$ 于区域 $j$ 结束，表示为 $f^{j,\tau}\_{i,t}$。显然，交通流反映了区域间的连通性，也反映了个体的移动。图1（c）给出了流量和流动的展示。

**Problem(Traffic Volume Prediction)** 给定知道时段 $t$ 的数据，交通流预测问题目标是预测时段 $t+1$ 的起始和结束流量。

# Spatial-Temporal Dynamic Network*

这部分，我们讲一下细节。图1是我们模型的架构。

![Figure1](/images/revisiting-spatial-temporal-similarity-a-deep-learning-framework-for-traffic-prediction/Fig1.JPG)

## Local Spatial-Temporal Network

为了捕获时空序列依赖，在出租车需求预测上，融合局部 CNN 和 LSTM 展示出了非常好的表现。我们这里使用局部 CNN 和 LSTM 处理空间和短期时间依赖。为了手动地提升两种流量的预测（起始和结束），我们将他们集成起来。这部分称为 Local Spatial-Temporal Network (LSTN)。

**Local spatial dependency** 卷积神经网络用来捕获空间关系。将整个城市看作一张图片，简单地使用 CNN 不能获得最好的性能。包含了弱关系的区域会导致预测性能下降。因此，我们使用局部 CNN 对空间依赖建模。

对于每个时段 $t$，我们将目标区域 $i$ 和它周围的邻居看作是 $S \times S$ 大小的图片，两个通道 $\mathbf{Y}\_{i,t} \in \mathbb{R}^{S \times S \times 2}$。一个通道包含起始流量信息，另一个是结束流量信息。目标区域在图像的中间。局部 CNN 使用 $\mathbf{Y}\_{i,t}$ 作为输入 $\mathbf{Y}^{(0)}\_{i,t}$，每个卷积层的定义如下：

$$\tag{1}
\mathbf{Y}^{(k)}\_{i,t} = \text{ReLU}(\mathbf{W}^{(k)} \ast \mathbf{Y}^{(k-1)}\_{i,t} + \mathbf{b}^{(k)}),
$$

其中 $\mathbf{W}^{(k)}$ 和 $\mathbf{b}^{(k)}$ 是参数。堆叠 $K$ 层卷积后，用一个全连接来推测区域 $i$ 的空间表示，记为 $\mathbf{y}\_{i,t}$。

**Short-term Temporal Dependency** 我们使用 LSTM 捕获空间序列依赖。我们使用原始版本的 LSTM：

$$\tag{2}
\mathbf{h}\_{i,t} = \text{LSTM}([\mathbf{y}\_{i,t};\mathbf{e}\_{i,t}], \mathbf{h}\_{i,t-1}),
$$

其中，$\mathbf{h}\_{i,t}$ 是区域 $i$ 在时段 $t$ 的输出表示。$\mathbf{e}\_{i,t}$ 表示外部因素。因此，$\mathbf{h}\_{i,t}$ 包含空间和短期时间信息。

## Spatial Dynamic Similarity: Flow Gating Mechanism

局部 CNN 用于捕获空间依赖。CNN 通过局部连接和权重共享处理局部结构相似性。在局部 CNN 中，局部空间依赖依靠历史交通流量的相似度。然而，流量的空间依赖是平稳的，不能完全地反映目标区域和其邻居间的关系。一个直接的表示区域间关系的方式是交通流。如果两个区域间有很多流量，那么他们之间的关系强烈（也就是他们更相似）。交通流可以用于控制流量信息在区域间的转移。因此，我们设计了一个 Flow Gating Mechanism (FGM)，以层次的方式对动态空间依赖建模。

类似局部 CNN，我们构建了局部空间流量图来保护流量的空间依赖。一个时段和一个区域相关的流量分两种，流入和流出，两个流量矩阵可以如上构建，每个元素表示对应区域的流入流出流量。图1(c) 给了一个流出矩阵。

给定一个区域 $i$，我们获得过去 $l$ 个时段的相关的流量（即从 $t-l+1$ 到 $t$）。需要的流量矩阵拼接起来，表示为 $\mathbf{F}\_{i,t} \in \mathbb{R}^{S \times S \times 2l}$，$2l$ 是流量矩阵的数量。因为堆叠的流量矩阵包含了过去与区域 $i$ 相关的矩阵，我们使用 CNN 对区域间的空间流量关系建模，将 $\mathbf{F}\_{i,t}$ 作为输入 $\mathbf{F}^{(0)}\_{i,t}$。对于每层 $k$，公式如下：

$$\tag{3}
\mathbf{F}^{(k)}\_{i,t} = \text{ReLU}(\mathbf{W}^{(k)}\_f \ast \mathbf{F}^{(k-1)}\_{i,t} + \mathbf{b}^{(k)}\_f),
$$

其中 $\mathbf{W}^{(k)}\_f$ 和 $\mathbf{b}^{(k)}\_f$ 是参数。

每层，我们使用流量信息对区域间的动态相似性进行捕获，通过一个流量门限制空间信息。特别地，空间表示 $\mathbf{Y}^{i,k}\_t$ 作为每层的输出，受流量门调整。我们重写式1为：

$$\tag{4}
\mathbf{Y}^{(k)}\_{i,t} = \text{ReLU}(\mathbf{W}^{(k)} \ast \mathbf{Y}^{(k-1)}\_{i,t} + \mathbf{b}^{(k)}) \otimes (\mathbf{F}^{i,k-1}\_t),
$$

$\otimes$ 是element-wise product。

$K$ 个门控卷积层后，我们用一个全连接得到流量门控空间表示 $\mathbf{y}\_{i,t}$。

我们将式2中的空间表示 $\mathbf{y}\_{i,t}$ 替换为 $\hat{\mathbf{y}}\_{i,t}$。

## Temporal Dynamic Similarity: Periodically Shifted Attention Mechanism

在上面的局部时空网络中，只有前几个时段用于预测。然而，这会忽略长期依赖（周期），但周期在时空预测问题中又很重要。这部分我们考虑长期依赖。

训练 LSTM 来处理长期信息不是一个简单的任务，因为序列长度的增加导致梯度消失，因此会减弱周期性的影响。为了解决这个问题，预测目标的相对时段（即昨天的这个时候，前天这个时候）应该被建模。然而，单纯地融入相对时段是不充分的，会忽略周期的平移，即交通数据不是严格周期的。举个例子，周末的高峰通常发生在下午的后半段，不同的日子时间不一致，从4:30pm到6:00pm变化。周期的平移在交通序列中很普遍，因为事故或堵塞的发生。图 2a 和 2b 分别是不同的天和周的时间平移的例子。这两个时间序列是从 NYC 的出租车数据中算的从 Javits Center出发的流量。显然，交通序列是周期性的，但是这些序列的峰值（通过红圈标记）在不同的日子里时间不一样。此外，对比两张图，周期性不是严格按日或按周的。因此，我们设计了一个 Periodically Shifted Attention Mechanism (PSAM) 来解决问题。详细的方法如下。

我们专注于解决日周期的平移问题。如图 1(a) 所示，从前 $P$ 天获得的相对时段用来处理周期依赖。对于每天，为了解决时间平移问题，我们从每天中额外的选择 $Q$ 个时段。举个例子，如果预测的时间是9:00-9:30pm，我们选之前的一个小时和之后的一个小时，即8:00-19:30pm，$\vert Q \vert = 5$。这些时段 $q \in Q$ 用来解决潜在的时间平移问题。此外，我们使用对每天 $p \in P$ 保护每天的序列信息，公式如下：

$$\tag{5}
\mathbf{h}^{p,q}\_{i,t} = \text{LSTM}([\mathbf{y}^{p,q}\_{i,t}; \mathbf{e}^{p,q}\_{i,t}], \mathbf{h}^{p,q-1}\_{i,t}),
$$

其中，$\mathbf{h}^{p,q}\_{i,t}$ 是对于区域 $i$ 的预测时间 $t$，时段 $q$ 在前一天 $p$ 的表示。

我们用了一个注意力机制捕获时间平移并且获得了前几天的每一天的一个表示。前几天每一天的表示 $\mathbf{h}^p\_{i,t}$ 是时段 $q$ 每一个选中时间的带权加和，定义为：

$$\tag{6}
\mathbf{h}^p\_{i,t} = \sum\_{q \in Q} \alpha^{p,q}\_{i,t} \mathbf{h}^{p,q}\_{i,t},
$$

权重 $\alpha^{p,q}\_{i,t}$ 衡量了在 $p \in P$ 这天时段 $q$ 的重要性。重要值 $\alpha^{p,q}\_{i,t}$ 通过对比从短期记忆（式2）得到的时空表示和前一个隐藏状态 $\mathbf{h}^{p,q}\_{i,t}$ 得到。权重定义为：

$$\tag{7}
\alpha^{p,q}\_{i,t} = \frac{\text{exp}(\text{score}(\mathbf{h}^{p,q}\_{i,t}, \mathbf{h}\_{i,t}))} {\sum\_{q \in Q} \text{exp}(\text{score} (\mathbf{h}^{p,q}\_{i,t}, \mathbf{h}\_{i,t})}
$$

类似 (Luong, Pham and Manning 2015)，注意力分数的定义可以看作是基于内容的函数：

$$\tag{8}
\text{score}(\mathbf{h}^{p,q}\_{i,t}, \mathbf{h}\_{i,t}) = \mathbf{v}^\text{T} \text{tanh} (\mathbf{W\_H} \mathbf{h}^{p,q}\_{i,t} + \mathbf{W\_X} \mathbf{h}\_{i,t} + \mathbf{b\_X}),
$$

其中，$\mathbf{W\_H}, \mathbf{W\_X}, \mathbf{b\_X}, \mathbf{v}$ 是参数，$\mathbf{v}^\text{T}$ 是转置。对于前面的每一天 $p$，我们得到一个周期表示 $\mathbf{h}^p\_{i,t}$。然后我们使用另一个 LSTM 用这些周期表示作为输入，保存序列信息，即

$$\tag{9}
\hat{\mathbf{h}}^p\_{i,t} = \text{LSTM}(\mathbf{h}^p\_{i,t}, \hat{\mathbf{h}}^{p-1}\_{i,t}).
$$

我们将最后一个时段的输出 $\hat{\mathbf{h}}^P\_{i,t}$ 看作是时间动态相似度的表示（即长期周期信息）。

## Joint Training

我们拼接把短期表示 $\mathbf{h}\_{i,t}$ 和 长期表示 $\hat{\mathbf{h}}^P\_{i,t}$ 拼接得到 $\mathbf{h}^c\_{i,t}$，对于预测区域和时间来说既保留了短期依赖又保留了长期依赖。我们将 $\mathbf{h}^c\_{i,t}$ 输入到全连接中，获得每个区域 $i$ 流入和流出流量的最终预测值，分别表示为 $y^i\_{s,t+1}$ 和 $y^i\_{e,t+1}$。最终预测函数定义为：

$$\tag{10}
[y^i\_{s,t+1}, y^i\_{e,t+1}] = \text{tanh}(\mathbf{W}\_{fa} \mathbf{h}^c\_{i,t} + \mathbf{b}\_{fa}),
$$

因为我们做了归一化，所以输出的范围在 $(-1, 1)$，输出值会映射回需求值。

我们同时预测出发和到达流量，损失函数定义为：

$$\tag{11}
\mathcal{L} = \sum^n\_{i=1} \lambda (y^s\_{i,t+1} - \hat{y}^s\_{i, t+1})^2 + (1 - \lambda) (y^e\_{i,t+1} - \hat{y}^e\_{i, t+1})^2,
$$

$\lambda$ 用来平衡流入和流出的影响。区域 $i$ 在时间 $t+1$ 实际的流入和流出流量表示为 $\hat{y}^s\_{i, t+1}, \hat{y}^e\_{i, t+1}$。

# Experiment

## Experiment Settings

**Datasets** 我们在两个 NYC 的大型数据集上评价了模型。每个数据集包含旅行记录，详情如下：

- NYC-Taxi：2015 年 22349490 个出租车的旅行记录，从 2015年1月1日到2015年3月1日。实验中，我们使用1月1日到2月10日作为训练，剩下20天测试。
- NYC-Bike：2016 年 NYC 自行车轨迹数据，7月1日到8月29日，包含了 2605648 条记录。前 40 天用来训练，后 20 天做测试。

**Preprocessing** 我们将整个城市分成 $10 \times 20$ 个区域。每个区域大小是 $1km \times 1km$。时段长度设为 30min。使用最大最小归一化 volume 和 flow 到 $[0, 1]$。预测后，使用逆变换后的值评价。我们使用滑动窗采样。测试模型的时候，滤掉 volumn 小于 10 的样本，这是工业界和学界常用的技巧 (Yao et al. 2018)。因为真实数据集中，关注较小的交通数据没有太大的意义。我们选择 80% 的数据训练，20% 验证。

**Evaluation Metric & Baselines** 两个指标：MAPE, RMSE。baselines：HA, ARIMA, Ridge, Lin-UOTD (Tong el al. 2017), XGBoost, MLP, ConvLSTM, DeepSD, ST-ResNet, DMVST-Net。

**Hyperparameter Settings** 我们基于验证集设定超参数。对于空间信息，64 个卷积核，$3 \times 3$ 大小。每个邻居的大小设定为 $7 \times 7$。层数 $K = 3$，对于 flow 考虑的时间跨度 $l = 2$。时间信息，短期 LSTM 长度为 7，长期周期信息 $\vert P \vert = 3$，周期平移注意力机制 $\vert Q \vert = 3$，LSTM 中隐藏表示的维度是 128。STDN 通过 Adam 优化，batch size 64，学习率 0.001。LSTM 中的 dropout 0.5。$\lambda$ 取 0.5。

## Results

**PerformanceComparison** 表 1 展示了我们的方法对比其他方法在两个数据集上的结果。我们跑每个 baseline 10次，取了平均和标准差。此外，我们也做了 t 检验。我们的方法在两个数据集上指标都很好。