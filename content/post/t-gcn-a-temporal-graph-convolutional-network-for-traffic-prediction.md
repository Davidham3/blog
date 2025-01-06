---
categories:
- 论文阅读笔记
date: 2019-03-07 09:03:16+0000
draft: false
math: true
tags:
- deep learning
- Spatial-temporal
- graph convolutional network
- Graph
- Time Series
title: 'T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction'
---
T-GCN，arxiv上面的一篇文章，用 GCN 对空间建模，GRU 对时间建模，很简单的模型。没有对比近几年的图卷积在时空数据挖掘中的模型。原文地址：[T-GCN: A Temporal Graph ConvolutionalNetwork for Traffic Prediction](https://arxiv.org/abs/1811.05320)

<!--more-->

# Abstract

精确和实时的交通预测在智能交通系统中扮演着重要的角色，对城市交通规划、交通管理、交通控制起着重要的作用。然而，交通预测由于其受限于城市路网且随时间动态变化，即有着空间依赖与时间依赖，早已成为一个公开的科学研究问题。为了同时捕获空间和时间依赖，我们提出了一个新的神经网络方法，时间图卷积网络模型 （T-GCN），将图卷积和门控循环单元融合起来。GCN 用来学习复杂的拓扑结构来捕获空间依赖，门控循环单元学习交通数据的动态变化来捕获时间依赖。实验表明我们的 T-GCN 模型比之前的方法要好。我们的 tf 实现：[代码仓库地址](https://github.com/lehaifeng/T-GCN)。

# 1 Introduction

随着智能交通系统的发展，交通预测受到了越来越多的关注。交通预测是高级交通管理系统中的关键部分，是实现交通规划、交通管理、交通控制的重要部分。交通预测是分析城市路网上交通状况、包括流量、车速、密度，挖掘交通模式，对路网上交通进行预测的一个过程。交通预测不仅能给管理者提供科学依据来预测交通拥挤并提前限制出行，还可以给旅客提供适当的出行路线并提高交通效率。然而，交通由于其空间和时间的依赖至今还是一个有难度的挑战：

（1）空间依赖。流量的改变主要受路网的拓扑结构控制。上游道路的交通状态通过转移影响下游的道路，下游的交通状态会通过反馈影响上游的状态。如图 1 所示，由于邻近道路的强烈影响，短期相似性从状态 1 （上游与中游相似）转移到 状态 2（上游与下游相似）。

（2）时间依赖。流量随时间动态改变，主要会出现周期性和趋势。如图 2（a）所示，路 1 的流量在一周内展示出了周期性变化。图 2（b）中，一天的流量也发生变换；举个例子，流量会被其前一时刻或更前的时刻的交通状况所影响。

![Figure1](/images/t-gcn-a-temporal-graph-convolutional-network-for-traffic-prediction/Fig1.JPG)
![Figure1](/images/t-gcn-a-temporal-graph-convolutional-network-for-traffic-prediction/Fig2.JPG)

有很多交通预测方法，一些考虑时间依赖，包括 ARIMA，Kalman filtering model, SVR, knn, Beyesian model, partial neural network model.上述方法考虑交通状况在时间上的动态变化，忽略了空间依赖，导致不能精确预测。为了更好地刻画空间特征，一些研究引入了卷积神经网络对空间建模；然而，卷积适用于欧氏空间的数据，如图像、网格等。这样的模型不能在城市路网这样有着复杂拓扑结构的环境下工作，所以他们不能描述空间依赖。

为了解决这个问题，我们提出了新的交通预测方法，时间图卷积网络 （T-GCN），用于对基于城市路网的交通预测任务。我们的贡献有三点：

（1） 我们提出的模型结合了 GCN 和 GRU，图卷积捕获路网的拓扑结构做空间建模，GRU 捕获路网上交通数据的时间依赖。T-GCN 模型可以用于其他时空预测任务上。

（2） T-GCN 的预测结果比其他的方法好，表明我们的 T-GCN 模型不仅可以做短期预测，也可以做长期预测。

（3）我们使用深圳市罗湖区的出租车速度数据和洛杉矶线圈数据。结果表明我们的预测误差比所有的 baseline 小了 1.5%到57.8%，表明 T-GCN 在交通预测上的优越性。

# 2 Related work

智能交通系统交通预测是现在的一个重要研究问题。现存的方法分两类：模型驱动的方法和数据驱动的方法。首先，模型驱动的方法主要解释交通流量、速度、密度的瞬时性和平稳性。这样的方法需要基于先验知识的系统建模。代表方法包括排队论模型，细胞传递模型，交通速度模型，microscopic fundamental diagram model 等等。实际中，交通数据受多种因素影响，很难获得一个精准的交通模型。现存的模型不能精确地描述复杂的现实环境中的交通数据的变化。此外，这些模型的构建需要很强的计算能力，而且很容易收到交通扰乱和采样点空间等问题的影响。

数据驱动的方法基于数据的规律性，从统计学推测变化局势，然后用于预测。这类方法不分析物理性质和交通系统的动态行为，有很高的灵活性。早期的方法包括历史均值模型，使用历史周期的交通流量均值作为预测值。这个方法不需要假设，计算简单而且还快，但是不能有效地拟合时间特征，预测的精准度低。随着研究的深入，很多高精度的方法涌现出来，主要分为参数模型和非参数模型。

参数模型提前假设回归函数，参数通过对原始数据处理得到，基于回归函数对交通流预测。时间序列模型，线性回归模型，Kalman filtering model 是常用的方法。时间序列模型将观测到的时间序列拟合进一个模型，然后用来预测。早在 1976 年，Box and Jenkins 提出了 ARIMA，Hamed 等人使用 ARIMA 预测城市内的交通流量。为了提高模型的精度，不同的变体相继被提出，Kohonen ARIMA，subset ARIMA，seasonal ARIMA 等等。Lippi 等人对比支持向量回归和 seasonal ARIM，发现 SARIMA 模型在交通拥堵上的预测有更好的结果。线性回归模型基于历史数据构建模型来预测。2004 年，Sun 等人使用 local linear model 解决了区间预测，在真实数据集上获得了较好的效果。Kalman filtering model 基于前一时刻和当前时刻的交通状态预测未来的状态。1984 年，Okutani 等人使用 Kalman filtering 理论建立了交通流状态预测模型。后续，一些研究使用 Kalman filtering 模型解决交通预测任务。

传统的参数模型算法简单，计算方便。然而，这些方法依赖平稳假设，不能反映交通数据的非线性和不确定性，也不能克服交通事件这种随机性事件。非参数模型很好地解决这些问题，只需要足够的历史信息能自动地从中学到统计规律即可。常见的非参数模型包括：k近邻，支持向量回归，Fuzzy Logic 模型等。

近些年，随着深度学习的快速发展，深度神经网络可以捕获交通数据的动态特征，获得很好的效果。根据是否考虑空间依赖，模型可以划分成两类。一些方法只考虑时间依赖，如 Park 等人使用 FNN 预测交通流。Huang 等人使用深度置信网络 DBN 和回归模型在多个数据集上证明可以捕获交通数据中的随机特征，提升预测精度。此外，RNN 及其变体 LSTM, GRU 可以有效地使用自循环机制，他们可以很好地学习到时间依赖并获得更好的预测结果。

这些模型考虑时间特征但是忽略空间依赖，所以交通数据的变化不受城市路网的限制，因此他们不能精确的预测路上的交通状态。解决交通预测问题的关键是充分利用空间和时间依赖。为了更好的刻画空间特征，很多研究已经在这个基础上进行了提升。Lv 等人提出了一个 SAE 模型从交通数据中捕获时空特征，实现短期交通流的预测。Zhang 等人提出了一个叫 ST-ResNet 的模型，基于人口流动的时间近邻、周期和趋势这些特征设计了残差卷积网络，然后三个网络和外部因素动态地聚合起来，预测城市内每个区域人口的流入和流出。Wu 等人设计了一个特征融合架构通过融合 CNN 和 LSTM 进行短期预测。一个一维的 CNN 用于捕获空间依赖，两个 LSTM 用来挖掘交通流的短期变化和周期性。Cao 等人提出一个叫 ITRCN 的端到端模型，将交互的网络交通转换为图像，使用 CNN 捕获交通的交互式功能，用 GRU 提取时间特征，预测误差比 GRU 和 CNN 分别高了 14.3% 和 13.0%。Ke 等人提出一个新的深度学习方法叫融合卷积长短时记忆网络（FCL-Net），考虑空间依赖、时间依赖，以及异质依赖，用于短期乘客需求预测。Yu 等人用深度卷积神经网络捕获空间依赖，用 LSTM 捕获时间动态性，在北京交通网络数据上展示出了 SRCN 的优越性。

尽管上述方法引入了 CNN 对空间依赖建模，在交通预测任务上有很大的进步，但 CNN 本质上只适用于欧氏空间，在有着复杂拓扑结构的交通网络上不能刻画空间依赖。因此，这类方法有缺陷。近些年，图卷积网络的发展，可以用来捕获图网络的结构特征，提供更好的解决方案。Li 等人提出了 DCRNN 模型，通过图上的随机游走捕获空间特征，通过编码解码结构捕获时间特征。

基于这个背景，我们提出了新的神经网络方法捕获复杂的时空特征，可以用于基于城市路网的交通预测任务上。

# 3 Methodology

## 3.1 Problem Definition

目标是基于历史信息预测未来。我们的方法中，交通信息是一个通用的概念，可以是速度、流量、密度。我们在实验的时候将交通信息看作是速度。

定义1：路网 $G$。我们用图 $G = (V, E)$ 描述路网的拓扑结构，每条路是一个顶点，$V$ 顶点集，$V = \lbrace v\_1, v\_2, \dots, v\_N \rbrace$，$N$ 是顶点数，$E$ 是边集。邻接矩阵 $A$ 表示路的关系，$A \in R^{N \times N}$。邻接矩阵只有 0 和 1。如果路之间有连接就为 1， 否则为 0。

定义2：特征矩阵 $X^{N \times P}$。我们将交通信息看作是顶点的特征。$P$ 表示特征数，$X\_t \in R^{N \times i}$ 用来表示时刻 $i$ 每条路上的速度。

时空交通预测的问题可以看作学习一个映射函数：

$$\tag{1}
[X\_{t+1}, \dots, X\_{t+T}] = f(G; (X\_{t-n}, \dots, X\_{t-1}, X\_t))
$$

$n$ 是历史时间序列的长度，$T$ 是需要预测的长度。

## 3.2 Overview

T-GCN 模型有两个部分：GCN 和 GRU。图 3 所示，我们使用历史 $n$ 个时刻的时间序列数据作为输入，图卷积网络捕获路网拓扑结构获取空间依赖。然后将带有空间特征的时间序列放入 GRU 中，通过信息在单元间的传递捕获动态变化，获得时间特征。最后，将结果送入全连接层。

![Figure3](/images/t-gcn-a-temporal-graph-convolutional-network-for-traffic-prediction/Fig3.JPG)

## 3.3 Methodology

### 3.3.1 Spatial Dependence Modeling

获取复杂的空间依赖在交通预测中是一个关键问题。传统的 CNN 只能用于欧氏空间。城市路网不是网格，CNN 不能反映复杂的拓扑结构。GCN 可以处理图结构，已经广泛应用到文档分类、半监督学习、图像分类中。GCN 在傅里叶域中构建滤波器，作用在顶点及其一阶邻居上，捕获顶点间的空间特征，可以通过堆叠构建 GCN 模型。如图 4 所示，假设顶点 1 是中心道路，GCN 模型可以获取中心道路和它周围道路的拓扑关系，将这个结构和道路属性编码，获得空间依赖。总之，我们用 GCN 模型从交通数据中学习空间特征。两层 GCN 表示为：

$$\tag{2}
f(X, A) = \sigma(\hat{A} Relu(\hat{A} X W\_0) W\_1)
$$

$\hat{A} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$ 表示预处理，$\tilde{A} = A + I\_N$ 表示加了自连接的邻接矩阵。

![Figure4](/images/t-gcn-a-temporal-graph-convolutional-network-for-traffic-prediction/Fig4.JPG)

### 3.3.2 Temporal Dependence Modeling

因为 GRU 比 LSTM 参数少，训练快，我们使用 GRU 获取交通数据的时间依赖。如图 5 所示，$h\_{t-1}$ 表示时刻 $t-1$ 的隐藏状态；$x\_t$ 表示时刻 $t$ 的交通信息；$r\_t$ 表示重置门，用来控制忽略前一时刻信息的程度；$u\_t$ 是更新门，用来控制将信息从上一时刻拿到这个时刻的程度；$c\_t$ 是时刻 $t$ 的记忆内容；$h\_t$ 是时刻 $t$ 的输出状态。GRU 通过将时刻 $t-1$ 的隐藏状态和当前时刻的交通信息作为输入，获取时刻 $t$ 的交通状态。在捕获当前时刻的交通信息的时候，模型仍保留着历史信息，且有能力捕获时间依赖。

![Figure5](/images/t-gcn-a-temporal-graph-convolutional-network-for-traffic-prediction/Fig5.JPG)

### 3.3.3 Temporal Graph Convolutional Network

为了同时从交通数据中捕获时空依赖，我们提出了时间图卷极网络（T-GCN）。如图6所示，左侧是时空交通预测的过程，右侧是一个 T-GCN 细胞的结构，$h\_{t-1}$ 表示 $t-1$ 时刻的输出，GC 是图卷积过程，$u\_t, r\_t$ 是时刻 $t$ 的更新门和重置门，$h\_t$ 表示时刻 $t$ 的输出。计算过程如下。$f(A, X\_t)$ 表示图卷积过程，如式 2 定义。$W$ 和 $b$ 表示训练过程的权重与偏置。

$$\tag{3}
u\_t = \sigma(W\_u[f(A, X\_t), h\_{t-1}] + b\_u)
$$

$$\tag{4}
r\_t = \sigma(W\_r[f(A, X\_t), h\_{t-1}] + b\_r)
$$

$$\tag{5}
c\_t = tanh(W\_c[f(A, X\_t), (r\_t \ast h\_{t-1})] + b\_c)
$$

$$\tag{6}
h\_t = u\_t \ast h\_{t-1} + (1 - u\_t) \ast c\_t
$$

总之，T-GCN 能处理复杂的空间依赖和时间动态性。

![Figure6](/images/t-gcn-a-temporal-graph-convolutional-network-for-traffic-prediction/Fig6.JPG)

### 3.3.4 Loss Function

损失函数如式 7。第一项用来减小速度的误差。第二项 $L\_{reg}$ 是一个 $L2$ 正则项，避免过拟合，$\lambda$ 是超参。

$$\tag{7}
loss = \Vert Y\_t - \hat{Y}\_t \Vert + \lambda L\_{reg}
$$

# 4 Experiments

## 4.1 Data Description

两个数据集，深圳出租车和洛杉矶线圈。两个数据集都和车速有关。

（1）SZ-taxi。数据是2015年1月1日到1月31日的深圳出租车轨迹数据。我们选了罗湖区 156 个主要路段作为研究区域。实验数据主要有两部分。一个是 156 * 156 的邻接矩阵，另一个是特征矩阵，描述了速度随时间的变化。我们将速度以 15 分钟为单位聚合。

（2）Los-loop。数据集是洛杉矶县高速公路线圈的实时数据。我们选了 207 个监测器，数据是 2012年5月1日到5月7日的数据。我们以5分钟为单位聚合车速。数据也是一个邻接矩阵和一个特征矩阵。我们用线性插值填补了缺失值。

我们将输入数据归一化到 $[0, 1]$。此外，80% 的数据用来训练，20% 用来测试。我们预测未来15、30、45、60分钟的车速。

## 4.2 Evaluation Metrics

（1）RMSE:

$$\tag{8}
RMSE = \sqrt{\frac{1}{n} \sum^n\_{i=1} (Y\_t - \hat{Y}\_t)^2}
$$

（2）MAE:

$$\tag{9}
MAE = \frac{1}{n} \sum^n\_{i=1} \vert Y\_t - \hat{Y}\_t \vert
$$

（3）Accuracy:

$$\tag{10}
Accuracy = 1 - \frac{\Vert Y - \hat{Y} \Vert}{\Vert Y \Vert\_F}
$$

（4）Coefficient of Determination (R2):

$$\tag{11}
R^2 = 1 - \frac{\sum\_{i=1} (Y\_t - \hat{Y}\_t)^2}{\sum\_{i=1}(Y\_t - \bar{Y})^2}
$$

（5）Explained Variance Score(Var):

$$
var = 1 - \frac{Var\lbrace Y - \hat{Y}\rbrace}{Var\lbrace Y\rbrace}
$$

RMSE 和 MAE 用来评估预测误差：越小越好。精度衡量预测的精度：越大越好。$R^2$ 和 Var 计算相关系数，评估预测结果表达真实数据的能力，越大越好。

## 4.3 Model Parameters Designing

(1) Hyperparameter

学习率、batch size、训练论述，隐藏层数。我们设定的是学习率0.001，batch size 64，轮数 3000 轮。

隐层单元数对 T-GCN 来说是个重要的参数，因为不同的单元数可能会影响预测精度。我们通过实验选取了最优的隐藏单元数。

看不下去了。。。

![Figure7](/images/t-gcn-a-temporal-graph-convolutional-network-for-traffic-prediction/Fig7.JPG)

![Table1](/images/t-gcn-a-temporal-graph-convolutional-network-for-traffic-prediction/Table1.JPG)