---
categories:
- 论文阅读笔记
date: 2019-02-28 21:12:58+0000
description: AAAI 2019，滴滴的网约车需求预测，5个点预测1个点。空间依赖建模上：以图的形式表示数据，从空间地理关系、区域功能相似度、区域交通连通性三个角度构造了三个不同的图，提出了多图卷积，分别用
  k 阶 ChebNet 对每个图做图卷积，然后将多个图的卷积结果进行聚合(sum, average 等)成一个图；时间依赖建模上：提出了融合背景信息的 Contextual
  Gated RNN (CGRNN)，用 ChebNet 对每个结点卷积后，得到他们的邻居表示，即每个结点的背景信息表示，与原结点特征拼接，用一个两层全连接神经网络计算出
  T 个权重，将权重乘到历史 T 个时刻的图上，对历史值进行缩放，然后用一个共享的 RNN，针对每个结点形成的长度为 T 的时间序列建模，得到每个结点新的时间表示。最后预测每个点的网约车需求。原文地址：[Spatiotemporal
  Multi-Graph Convolution Network for Ride-hailing Demand Forecasting](http://www-scf.usc.edu/~yaguang/papers/aaai19_multi_graph_convolution.pdf)
draft: false
math: true
tags:
- deep learning
- Spatial-temporal
- graph convolutional network
- Graph
- Time Series
title: Spatiotemporal Multi-Graph Convolution Network for Ride-hailing Demand Forecasting
---
AAAI 2019，滴滴的网约车需求预测，5个点预测1个点。空间依赖建模上：以图的形式表示数据，从空间地理关系、区域功能相似度、区域交通连通性三个角度构造了三个不同的图，提出了多图卷积，分别用 k 阶 ChebNet 对每个图做图卷积，然后将多个图的卷积结果进行聚合(sum, average 等)成一个图；时间依赖建模上：提出了融合背景信息的 Contextual Gated RNN (CGRNN)，用 ChebNet 对每个结点卷积后，得到他们的邻居表示，即每个结点的背景信息表示，与原结点特征拼接，用一个两层全连接神经网络计算出 T 个权重，将权重乘到历史 T 个时刻的图上，对历史值进行缩放，然后用一个共享的 RNN，针对每个结点形成的长度为 T 的时间序列建模，得到每个结点新的时间表示。最后预测每个点的网约车需求。原文地址：[Spatiotemporal Multi-Graph Convolution Network for Ride-hailing Demand Forecasting](http://www-scf.usc.edu/~yaguang/papers/aaai19_multi_graph_convolution.pdf)

<!--more-->

# Abstract

区域级别的预测是网约车服务的关键任务。精确地对网约车需求预测可以指导车辆调度、提高车辆利用率，减少用户的等待时间，减轻交通拥堵。这个任务的关键在于区域间复杂的时空依赖关系。现存的方法主要关注临近区域的欧式关系建模，但是在距离较远的区域间组成的非欧式关系对精确预测也很关键。我们提出了 *spatiotemporal multi-graph convolution network* (ST-MGCN)。我们首先将非欧的关系对编码到多个图中，然后使用 multi-graph convolution 对他们建模。为了在时间建模上利用全局的背景信息，我们提出了 *contextual gated recurrent neural network*，用一个注意背景的门机制对不同的历史观测值重新分配权重。在两个数据集上比当前的 state-of-the-art 强 10%。

# Introduction

我们研究的问题是区域级别网约车需求预测，是智能运输系统的重要部分。目标是通过历史观测值，预测一个城市里面各区域未来的需求。任务的挑战是复杂的时空关系。一方面，不同区域有着复杂的依赖关系。举个例子，一个区域的需求通常受其空间上临近的区域所影响，同时与有着相同背景的较远的区域有联系。另一方面，非线性的依赖关系也存在于不同的时间观测值之间。预测一个时刻通常和多个历史的观测值相关，比如一小时前、一天前、甚至一周前。

![Figure1](/blog/images/spatiotemporal-multi-graph-convolution-network-for-ride-hailing-demand-forecasting/Fig1.JPG)

最近在深度学习的进步使得对基于区域级别的时空关系预测有了很好的结果。使用卷积神经网络和循环神经网络，得到了很多非常好的效果(Shi et al. 2015; Yu et al. 2017; Shi et al. 2017; Zhang, Zheng, and Qi 2017; Zhang et al. 2018a; Ma et al. 2017; Yao et al. 2018b; 2018a)。尽管有了很好的效果，但是我们认为在对时空关系建模上有两点被忽略了。其一，这些方法主要对不同区域的欧式关系建模，但是我们发现非欧关系很重要。图 1 是一个例子，对于区域 1，以及邻居区域 2，可能和很远的区域 3 有相似的功能，也就是他们都靠近学校和医院。此外，区域 1 还可能被区域 4 影响，区域 4 是通过高速公路直接与区域 1 相连的。其二：这些方法中，在使用 RNN 对时间关系建模时，每个区域是独立处理的，或者只基于局部信息。然而，我们认为全局和背景信息也很重要。举个例子，网约车需求的一个全局性的增长/减小通常表明一些可能会影响未来需求的活动发生了。

我们提出了 ST-MGCN 解决这些问题。在 ST-MGCN 中，我们提出了将区域间非欧关系编码进多个图的方法。不同于 Yao et al. 2018b 给每个区域使用图嵌入作为额外的不变特征，我们用图卷积对区域间的关系对直接建模。图卷积在预测的时候可以聚合邻居特征，传统的图嵌入难以做到这一点。此外，在对时间关系建模时，为了聚合全局的背景信息，我们提出了 contextual gated recurrent neural network (CGRNN)。通过一个基于全局信息计算的门机制增强 RNN，对不同时间步的观测值重新赋权重。我们在两个大型的真实数据集上做了测试，ST-MGCN 比 baselines 好了一大截。我们主要的贡献是：

- 识别了网约车需求预测中的非欧关系，将他们编码进多个图。利用多图卷积对这些关系建模。
- 对时间依赖，提出了 Contextual Gated RNN (CGRNN) 来集成全局背景信息。
- 在两个大型真实数据集上做了实验，提出的方法比 state-of-the-art 在相对误差上小了 10%.

# Related work

## Spatiotemporal prediction in urban computing

时空预测是数据驱动的城市管理的基础问题。有很多关于这方面的工作，自行车流量预测(Zhang, Zheng, and Qi 2017)，出租车需求(Ke et al. 2017b; Yao et al. 2018b)，到达时间(Li et al. 2018b)，降雨量(Shi et al. 2015; 2017)，对矩形区域的聚合值进行预测，区域关系通过地理距离建模。具体来讲，城市数据的空间结构通过矩阵形式表示，每个元素表示一个矩形区域。在之前的工作中，区域和他们的关系对一般表示成欧式结构，使得卷积神经网络可以有效地利用这个结构来预测。

非欧结构的数据也存在于城市计算。通常，基于站点或点的预测任务，像流量预测(Li et al. 2018c; Yu, Yin, and Zhu 2018; Yao et al. 2018a)，基于点的出租车需求预测(Tong et al. 2017)以及基于站点的自行车流量预测(Chai, Wang, and Yang 2018)是很自然的非欧结构，数据不再是矩阵形式，卷积神经网络也不那么有效了。人工定制的特征工程或图卷积网络是处理非欧结构数据目前最好的方法。不同于之前的工作，ST-MGCN 将区域间的关系对编码进语义图中。尽管 ST-MGCN 是对基于区域的预测设计的，但是区域关系的非规整性使得它实际是对非欧数据进行预测。

在 (Yao et al. 2018b)，作者提出 DMVST-Net，将区域间关系编码进图中来预测出租车需求。DMVST-Net 主要使用图嵌入作为额外特征来预测，没有使用相关区域的需求值（目标值）。在 (Yao et al. 2018a) 的工作中，作者通过注意力机制对周期性的平移问题建模提升了性能。但是，这些方法都没有直接对区域间的非欧关系建模。我们的工作中，ST-MGCN 使用提出的多图卷积从相关区域聚合特征，从不同角度的相关区域的预测值中做预测。

最近在对帕金森的神经图像分析 (Zhang et al. 2018b) 的研究中，图卷积在空间特征提取上很有效。他们使用 GCN 从最相似的区域中学习特征，提出了多视图结构融合了不同的 MRI。然而，上述工作没有考虑时间依赖。ST-GCN 用于基于骨骼的动作识别(Li et al. 2018a; Yan, Xiong, and Lin 2018)。ST-GCN 的变换是一个空间依赖和局部时间循环的组合。然而，我们认为这些模型，在时间依赖建模上，背景信息或全局信息被忽略了。

## Graph convolution network

图卷积网络定义在图 $\mathcal{G} = (V, \boldsymbol{A})$ 上，$V$ 是顶点集，$\boldsymbol{A} \in \mathbb{R}^{\vert V \vert \times \vert V \vert}$ 是邻接矩阵，元素表示顶点间是否相连。GCN 可以用不同的感受野从不同的非欧结构中提取局部特征(Hammond et al. 2011)。令 $\boldsymbol{L} = \boldsymbol{I} - \boldsymbol{D}^{-1/2} \boldsymbol{A} \boldsymbol{D}^{-1/2}$ 表示图拉普拉斯矩阵，$\boldsymbol{D}$ 是度矩阵，图卷积操作 (Defferrard, Bresson, and Vandergheynst 2016) 定义为：

$$
\boldsymbol{X}\_{l+1} = \sigma (\sum^{K-1}\_{k=0} \alpha\_k \boldsymbol{L}^k \boldsymbol{X}\_l),
$$

$\boldsymbol{X}\_l$ 表示第 $l$ 层的特征，$\alpha\_k$ 表示可学习的参数，$\boldsymbol{L}^k$ 是图拉普拉斯矩阵的 $k$ 次幂，$\sigma$ 是激活函数。

## Channel-wise attention

Channel-wise attention (Hu, Shen, and Sun 2018; Chen et al. 2017) 在 cv 的论文中提出。本质是给每个通道学习一个权重，为了找到最重要的帧，然后基于他们更高的权重。$\boldsymbol{X} \in \mathbb{R}^{W \times H \times C}$ 表示输入，$W$ 和 $H$ 是输入图像的维度，$C$ 表示通道数，channel-wise attention 计算方式如下：

$$\tag{1}
z\_c = F\_{pool}(\boldsymbol{X}\_{:,:,c}) = \frac{1}{WH} \sum^W\_{i=0} \sum^H\_{j=0} X\_{i,j,c} \quad \text{for} c=1,2,\dots,C \\
\boldsymbol{s} = \sigma(\boldsymbol{W}\_2 \delta (\boldsymbol{W}\_1 \boldsymbol{z})) \\
\tilde{\boldsymbol{X}}\_{:,:,c} = \boldsymbol{X}\_{:,:,c} \circ s\_c \quad \text{for} c=1,2,\dots,C
$$

$F\_{pool}$ 是全局池化操作，把每个通道聚合成一个标量 $\boldsymbol{z}\_c$，$c$ 是通道的下标。用一个注意力机制对聚合的向量 $\boldsymbol{z}$ 使用非线性变换生成自适应的通道权重 $\boldsymbol{s}$，$\boldsymbol{W}\_1, \boldsymbol{W}\_2$ 是对应的权重，$\delta, \sigma$ 是 ReLU 和 sigmoid 激活函数。$\boldsymbol{s}$ 通过矩阵乘法乘到输入上。最后，输入通道基于学习到的权重得到了缩放。我们使用这个方法，针对一系列的图生成了时间依赖的注意力分数。

# Methodology

## Region-level ride-hailing demand forecasting

我们将城市分为相同大小的网格，每个格子定义为一个区域 $v \in V$，$V$ 表示城市内所有不相交的区域。$\boldsymbol{X}^{(t)}$ 表示第 $t$ 个时段所有区域的订单。*区域级别的网约车需求预测* 问题定义为：给定一个定长的输入，对单个时间步进行时空预测，也就是学习一个函数 $f: \mathbb{R}^{\vert V \vert \times T} \rightarrow \mathbb{R}^{\vert V \vert}$，将所有区域的历史需求映射到下一个时间步上。

$$
[\boldsymbol{X}^{(t-T+1)}, \dots, \boldsymbol{X}^{(t)}] 
$$

**Framework overview** ST-MGCN 的系统架构如图2。我们从不同的角度表示区域间的关系，顶点表示区域，边对区域间的关系编码。首先，我们使用提出的 CGRNN，考虑全局背景信息对不同时间的观测值进行聚合。然后，使用多图卷积捕获区域间不同类型的关系。最后，使用全连接神经网络将特征映射到预测上。

![Figure2](/blog/images/spatiotemporal-multi-graph-convolution-network-for-ride-hailing-demand-forecasting/Fig2.JPG)

## Spatial dependency modeling

我们用图将区域间关系建模成三种类型，（1）邻居图 $\mathcal{G}\_N = (V, \boldsymbol{A}\_N)$，编码了空间相近程度，（2）功能相似度图 $\mathcal{G}\_F = (V, \boldsymbol{A}\_F)$，编码了区域的 POI 的相似度，（3）连接图 $\mathcal{G}\_T = (V, \boldsymbol{A}\_T)$，编码了距离较远的区域的连通性。我们的方法可以轻易地扩展到其他的图上。

**Neighborhood** 区域的邻居基于空间近邻程度定义。我们将 $3 \times 3$ 区域中的最中间的那个区域与他邻接的 8 个区域相连。

$$\tag{3}
A\_{N, ij} = \begin{cases}
1, \quad v\_i \quad \text{and} \quad v\_j \quad \text{are} \quad \text{adjacent}\\
0, \quad \text{otherwise}
\end{cases}
$$

**Functional similarity** 对一个区域做预测的时候，很自然的会想到和这个区域在功能上相似的区域会有帮助。区域功能可以由 POI 刻画，两个顶点间的边定义为 POI 的相似度：

$$\tag{3}
A\_{S,i,j} = \text{sim}(P\_{v\_i}, P\_{v\_j}) \in [0, 1]
$$

其中 $P\_{v\_i}, P\_{v\_j}$ 是区域 $v\_i$ 和 $v\_j$ 的 POI 向量，维度等于 POI 种类的个数，每个分量表示这个区域内这个 POI 类型的数量。

**Transportation connectivity** 运输系统也是一个重要因素。一般来说，这些空间距离上相距较远但是可以很方便到达的区域可以关联起来。这种连接包含高速公路、公路、地铁这样的公共运输。我们定义：如果两个区域间通过这些路直接相连，那么他们之间有边：

$$\tag{4}
A\_{C,i,j} = max(0, \text{conn}(v\_i, v\_j) - A\_{N,i,j}) \in \lbrace 0, 1\rbrace
$$

$\text{conn}(u, v)$ 表示 $v\_i$ 和 $v\_j$ 之间的连通性。邻居的边在这个图中移除掉了，减少冗余的关系，所以这个图最后是一个稀疏图。

**Multi-graph convolution for spatial dependency modeling** 有了这些图，我们提出了多图卷积对空间关系建模：

$$\tag{5}
\boldsymbol{X}\_{l+1} = \sigma(\bigsqcup\_{\mathbf{A} \in \mathbb{A}} f(\mathbf{A; \theta\_i}) \boldsymbol{X}\_l \mathbf{W}\_l)
$$

其中 $\boldsymbol{X}\_l \in \mathbb{R}^{\vert V \vert \times P\_l}, \boldsymbol{X}\_{l+1} \in \mathbb{R}^{\vert V \vert \times P\_{l+1}}$ 是第 $l$ 和 $l+1$ 层的特征向量，$\sigma$ 是激活函数，$\bigsqcup$ 表示聚合函数，如 sum, max, average etc. $\mathbb{A}$ 表示图的集合，$f(\mathbf{A}; \theta\_i) \in \mathbb{R}^{\vert V \vert \times \vert V \vert}$ 表示参数为 $\theta\_i$ 的基于图 $\mathbf{A} \in \mathbb{A}$ 的不同样本组成的矩阵的聚合值，$\mathbf{W}\_l \in \mathbb{R}^{P\_l \times P\_{l+1}}$ 表示特征变换矩阵，举个例子，如果 $f(\mathbf{A}, \theta\_i)$ 是拉普拉斯矩阵 $\mathbf{L}$ 的多项式，那么这就是多图上的 ChebNet。如果是 $\mathbf{I}$，那就是全连接神经网络。

我们实现的是 $K$ 阶 拉普拉斯 $\mathbf{L}$ 多项式，图 3 是一个中心区域通过图卷积层变换后的例子。假设邻接矩阵中的值不是 0 就是 1，$L^k\_{ij} \not = 0$ 表示 $v\_i$ 在 $k$ 步内可达 $v\_j$。根据卷积操作，$k$ 是空间特征提取时的感受野范围。使用图 1 的道路连通性图 $\mathcal{G}\_C = (V, \boldsymbol{A}\_C)$ 来说明。在邻接矩阵 $\boldsymbol{A}\_C$ 中，我们有：

$$
A\_{C,1,4} = 1; A\_{C,1,6} = 0; A\_{C,4,6} = 1,
$$

在 1 度拉普拉斯矩阵中对应的分量是：

$$
L^1\_{C,1,4} \not = 0; L^1\_{C,1,6} = 0; L^1\_{C,4,6} \not = 0
$$

![Figure3](/blog/images/spatiotemporal-multi-graph-convolution-network-for-ride-hailing-demand-forecasting/Fig3.JPG)

如果拉普拉斯矩阵的最大度数 $K$ 设为 $1$，那么区域 1 变换的特征向量，即 $\boldsymbol{X}\_{l+1, 1,:}$ 不会包含区域 6: $\boldsymbol{X}\_{l,6,:}$，因为 $L^1\_{C,1,6}=0$。当 $K$ 增大到 2 的时候，对应的元素 $L^2\_{C,1,6}$ 变成非零，$\boldsymbol{X}\_{l+1,1,:}$ 就可以利用 $\boldsymbol{X}\_{l,6,:}$ 的信息了。

基于多图卷积的空间依赖建模不限于上述三种图，可以轻易地扩展到其他的图上，适用于其他的时空预测问题上。多图卷积对区域间的关系进行特征提取。感受野小的时候，专注于近邻的区域。增大拉普拉斯阶数，或者堆叠卷积层可以增加感受野的范围，鼓励模型捕获全局依赖关系。

图嵌入是另一种对区域间关系建模的方法。在 DMVST-Net (Yao et al. 2018b)，作者使用图嵌入表示区域间关系，然后将嵌入作为额外特征加到每个区域上。我们认为 ST-MGCN 中的空间依赖建模方法比之前的方法好，因为：ST-MGCN 将区域间关系编码到图中，通过图卷积从相关区域聚合需求值。但是在 DMVST-Net 中区域关系是嵌入到一个基于区域的不随时间变化的特征中，作为的模型的输入，

尽管 DMVST-Net 也捕获了拓扑结构信息，但是它很难从相关的区域中通过区域关系聚合需求值。而且不变的特征对模型训练的贡献有限。

## Temporal correlation modeling

我们提出 Contextual Gated Recurrent Neural Network (CGRNN) 对不同时间步上的样本建模。CGRNN 通过使用一个上下文注意的门机制增强 RNN 将背景信息集成到时间建模中，结构如图 4。假设我们有 $T$ 个观测样本，$\boldsymbol{X}^{(t)} \in \mathbb{R}^{\vert V \vert \times P}$ 表示第 $t$ 个样本，$P$ 是特征数，如果特征只包含订单数，那就是 1。上下文门控机制如下：

$$tag{6}
\hat{\boldsymbol{X}}^{(t)} = [\boldsymbol{X}^{(t)}, F^{K'}\_\mathcal{G}(\boldsymbol{X}^{(t)})] \quad \text{for} \quad t = 1,2,\dots,T
$$

首先，上下文门控机制通过将临近区域的历史信息和当前区域拼接，得到了区域的描述信息。从相邻区域来的信息看作是环境信息，通过图卷积 $F^{K'}\_\mathcal{G}$ 使用最大阶数为 $K'$ 的拉普拉斯矩阵提取。上下文门控机制用来用图卷积操作集成临近区域的信息，然后使用一个池化：

$$\tag{7}
z^{(t)} = F\_{pool}(\hat{\boldsymbol{X}}^{(t)}) = \frac{1}{\vert V \vert} \sum^{\vert V \vert}\_{i=1} \hat{X}^{(t)}\_{i,:} \quad \text{for} \quad t=1,2,\dots,T
$$

然后，我们在所有的区域上使用全局平均池化 $F\_{pool}$ 生成每个时间步观测值的平均值。

$$tag{8}
\boldsymbol{s} = \sigma(\boldsymbol{W}\_2 \delta(\boldsymbol{W}\_1) \boldsymbol{z})
$$

然后使用一个注意力机制，$\boldsymbol{W}\_1, \boldsymbol{W}\_2$ 是参数，$\delta, \sigma$ 分别是 ReLU 和 sigmoid 激活。

$$\tag{9}
\tilde{\boldsymbol{X}^{(t)}} = \boldsymbol{X}^{(t)} \circ s^{(t)} \quad \text{for} \quad t=1,2,\dots,T
$$

最后，$\boldsymbol{s}$ 用来对每个时间样本进行缩放：

$$tag{10}
\boldsymbol{H}\_{i,:} = \text{RNN}(\tilde{\boldsymbol{X}}^{(1)}\_{i,:}, \dots, \tilde{\boldsymbol{X}}^{(T)}\_{i,:}; \boldsymbol{W}\_3) \quad \text{for} \quad i=1,\dots,\vert V \vert
$$

在上下文门控之后，使用一个共享的 RNN 对所有的区域进行计算，将每个区域聚合成单独的向量 $\boldsymbol{H}\_{i,:}$。使用共享 RNN 的原因是我们想找到一个对所有区域通用的聚合规则，这个规则鼓励模型泛化且减少模型的复杂度。

# Experiments

**Dataset** 北京和上海。时间是从2017年5月1日到2017年12月31日。5月1日到7月31日训练、8月1日到9月30日验证，剩下的测试。POI 数据是2017年的，包含13个类别。每个区域和一个 POI 向量相关，分量是这个 POI 类型在这个区域的个数。用来评估运输可达性的路网使用的是 OpenStreetMap (Haklay and Weber 2008)。

## Experimental Settings

学习任务是：$f: \mathbb{R}^{\vert V \vert \times T} \rightarrow \mathbb{R}^{\vert V \vert}$。实验中，我们将区域以 $1km \times 1km$ 的大小划分成网格。北京和上海分别 1296 和 896 个区域。就像 Zhang, Zheng, and Qi 2017 做的那样，网络的输入包含 5 个历史观测值，三个最近邻的部分，1个周期部分，一个最新的趋势部分。在构建运输可达性网络的时候，我们考虑了高速公路、公路、地铁。两个区域间只要有这样的路直接相连就认为是连通的。

$f(\mathbf{A}; \theta\_i)$ 选择的是 $K = 2$ 时的切比雪夫多项式，$\bigsqcup$ 是 sum 函数。隐藏层为3，每层 64 个隐藏单元，L2 正则，weight decay 是 $1e-4$。CGRNN 中的图卷积 $K'$ 是 1。

我们使用 ReLU 作为图卷积的激活函数。ST-MGCN 的学习率是 $2e-3$，使用验证集上的早停。所有的算法都用 tf 实现，adam 优化 RMSE。ST-MGCN 训练时用了 10G 内存，9G GPU 显存。在 Tesla P40 单卡上训练了一个半小时。

**Methods for evaluation** HA, LASSO, Ridge, Auto-regressive model(VAR, STAR), Gradient boosted machine (GBM), ST-ResNet (Zhang, Zheng and Qi 2017), DMVST-Net (Yao et al. 2018b), DCRNN, ST-GCN。

## Performance comparison

我们在验证集上用网格搜索调整了所有模型的参数，在测试集上跑了多次得到的最后的结果。我们使用 RMSE 和 MAPE 作为评价指标。表 1 展示了不同方法在 10 次以上的预测中的对比结果。

![Table1](/blog/images/spatiotemporal-multi-graph-convolution-network-for-ride-hailing-demand-forecasting/Table1.JPG)

我们在两个数据集上观测到了几个现象：（1）基于深度学习的方法能够对非线性的时空依赖关系建模，比其他的方法好；（2）ST-MGCN 在两个数据集上比其他的方法都好，比第二好的高出 10%；（3）对比其他的深度学习方法，ST-MGCN 的方差更小。

## Effect of spatial dependency modeling

为了研究空间和时间依赖建模的效果，我们通过减少模型中的组成部分评估了 ST-MGCN 的几个变体，包括：（1）邻居图，（2）功能相似性图，（3）运输连通性图。结果如表 2 所示。移除任何一个图都会造成性能损失，证明了每种关系的重要性。这些图编码了重要的先验知识，也就是区域间的相关性。

![Table2](/blog/images/spatiotemporal-multi-graph-convolution-network-for-ride-hailing-demand-forecasting/Table2.JPG)

为了评估集成多个区域关系的效果，我们扩展了基于单个图的模型，包括 DCRNN 和 STGCN，分别记为 DCRNN+ 和 ST-GCN+。结果如图 3，两个算法都得到了提升。

![Table3](/blog/images/spatiotemporal-multi-graph-convolution-network-for-ride-hailing-demand-forecasting/Table3.JPG)

## Effect of temporal dependency modeling

我们使用不同的方法对时间建模，评估 ST-GCN 对时间关系建模的效果。（1）平均池化：通过平均池化对历史观测值进行聚合，（2）RNN：使用 RNN 对历史观测值聚合，（3）CG：使用上下文门对不同的历史观测值赋权，不适用 RNN，（4）GRNN：不用图卷积的 CGRNN。结果如表 4。我们观察到了以下现象：
- 平均池化会盲目地平均不同的样本，导致性能下降，能做上下文依赖非线性时间聚合的 RNN 能显著地提升性能。
- CGRNN 增强了 RNN。移除 RNN 和 图卷积都导致性能下降，证明了每个部件的有效性。

![Table4](/blog/images/spatiotemporal-multi-graph-convolution-network-for-ride-hailing-demand-forecasting/Table4.JPG)

## Effect of model parameters

我们调整了两个最重要的参数来看不同参数对模型的影响，$K$ 和图卷积层数。图 5 展示了测试集上的结果。可以观察到随着层数的增加，错误先降后增。但是随着 $K$ 的增加，错误是先减小，后不变。越大的 $K$ 或层数使得模型能捕获全局关联性，代价是模型的复杂度会增加，更易过拟合。

![Figure5](/blog/images/spatiotemporal-multi-graph-convolution-network-for-ride-hailing-demand-forecasting/Fig5.JPG)

# Conclusion and Future work

我们研究的是网约车需求预测，要找寻这个问题唯一的时空依赖关系。我们提出的深度学习模型使用多个图对区域间的非欧关系建模，使用多图卷积明显的捕获了这个关系。然后用上下文门控机制增强了 RNN，在时间建模上集成了全局背景信息。在两个大型真实数据集上评估了模型，比 state-of-the-art好。未来的工作是：（1）在其他的时空预测任务上评估模型；（2）将提出的模型扩展到多步预测上。