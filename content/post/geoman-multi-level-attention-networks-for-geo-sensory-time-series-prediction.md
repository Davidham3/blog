---
categories:
- 论文阅读笔记
date: 2018-07-09 17:03:00+0000
description: 'IJCAI 2018，看了一部分，还没看完。原文链接：[GeoMAN: Multi-level Attention Networks for
  Geo-sensory Time Series Prediction](https://www.ijcai.org/proceedings/2018/476)'
draft: false
math: true
tags:
- deep learning
- Spatial-temporal
- Attention
- Time Series
title: 'GeoMAN: Multi-level Attention Networks for Geo-sensory Time Series Prediction'
---
IJCAI 2018，看了一部分，还没看完。原文链接：[GeoMAN: Multi-level Attention Networks for Geo-sensory Time Series Prediction](https://www.ijcai.org/proceedings/2018/476)
<!--more-->

# Abstract

大量的监测器被部署在各个地方，连续协同地监测周围的环境，如空气质量。这些检测器生成很多时空序列数据，之间有着空间相关性。预测这些时空数据很有挑战，因为预测受很多因素影响，比如动态的时空关联和其他因素。我们在这篇论文中使用多级基于注意力机制的 RNN 模型，结合空间、气象和检测器数据来预测未来的监测数值。我们的模型由两部分组成：
1. 多级注意力机制对时空依赖关系建模
2. 一个通用的融合模块对多领域的外部信息进行融合

实验用了两个真实的数据集，空气质量数据和水质监测数据，结果显示我们的模型比9个baselines都要好。

# 1 Introduction

现实世界中有大量的检测器，如空气监测站。每个监测站都有一个地理位置，不断地生成时间序列数据。一组检测器不断的监测一个区域的环境，数据间有空间依赖关系。我们成这样的监测数据为 *geosensory time series*。此外，一个检测器生成多种 geo-sensory 时间序列是很常见的，因为这个检测器同时监测不同的目标。举个例子，图1a，路上的循环检测器实时记录车辆通行情况，也记录他们的速度。图1b 展示了检测器每五分钟生成的关于水质的三个不同的气候指标。除了监测，对于 geo-sensory 时间序列预测还有一个重要的需求就是交通预测。

![Fig1](/images/geoman-multi-level-attention-networks-for-geo-sensory-time-series-prediction/Fig1.PNG)

然而，预测 geo-sensory 时间序列很有挑战性，主要受两个因素影响：
1. 动态时空关系
·检测器间复杂的关系。图1c展示了不同检测器的时间序列间的空间关系是高度动态的，随着时间不断改变。除此以外，geo-sensory时间序列根据地区有非线性的变化。当对动态关系建模时，传统方法（如概率图模型）的计算量会很大，因为他们有很多参数。
·检测器内部的动态关系。首先，一个geo-sensory时间序列通常由一种周期模式（如，图1c中的$S\_1$），这种模式一直变化，并且地理上也有改变。其次，检测器记录经常有很大的振动，很快地减少前一个数值的影响。因此，如何选择一个时间间隔来预测也是一个问题。
2. 外部因素。
检测器数据也被周围的环境影响，比如气象（例如强风），几点（比如早晚高峰）还有土地使用情况等。

为了解决这些挑战，我们提出了一个多级注意力网络（GeoMAN）来预测一个检测器未来几个小时的数值。我们的贡献有三点：
1. *多级注意力机制*。我们研发了一个多级注意力机制对动态时空关系建模。具体来说，第一级，我们提出了一个新的注意力机制，由局部空间注意力和全局空间注意力组成，用来捕获不同检测器时间序列间的复杂空间关系。第二级，时间注意力对一个时间序列中的动态时间关系进行建模。
2. *外部因素融合模型*。我们设计了一个通用融合模型融合不同领域的外部因素。学习到的隐含表示会输入至多级注意力网络中来提升这些外部因素的重要性。
3. *真实的评价*。我们基于两个真实的数据集评估我们的方法。大量的实验证明了我们的方法相比于baseline的优越性。

# 2 Preliminary

## 2.1 Notations

假设，有 $N\_g$ 个检测器，每个都生成 $N\_l$ 种时间序列。我们指定其中的一个为 *target series* 来预测，其它序列作为特征。时间窗为 $T$，我们使用 $\mathbf{Y} = (\mathbf{y}^1, \mathbf{y}^2, ..., \mathbf{y}^{N\_g}) \in \mathbb{R}^{N\_g \times T}$ 来表示所有目标序列在过去 $T$ 个小时的监测值，其中 $\mathbf{y}^i \in \mathbb{R}^T$ 属于监测器 $i$。我们使用 $\mathbf{X}^i = (\mathbf{x}^{i, 1}, \mathbf{x}^{i, 2}, ..., \mathbf{x}^{i, N\_l})^{\rm T} = (\mathbf{x}^i\_1, \mathbf{x}^i\_2, ..., \mathbf{x}^i\_T) \in \mathbb{R}^{N\_l \times T}$ 作为检测器 $i$ 的局部特征，其中 $\mathbf{x}^{i,k} \in \mathbb{R}^T$ 是这个检测器的第 $k$ 个时间序列，$\mathbf{x}^i\_t = (x^{i,1}\_t, x^{i,2}\_t, ..., x^{i,N\_l}\_t)^{\rm T} \in \mathbb{R}^{N\_l}$ 表示检测器 $i$ 在时间 $t$ 的所有时间序列的值。除了检测器 $i$ 的局部特征，由于不同检测器间的空间关系，其他的检测器会共享大量对于预测有用的信息。为了这个目的，我们将每个检测器的局部特征融合到集合 $\mathcal{X}^i = \lbrace \mathbf{X}^1, \mathbf{X}^2, ..., \mathbf{X}^{N\_g}\rbrace$ 中作为检测器 $i$ 的全局特征。

## 2.2 Problem Statement

给定每个检测器之前的值和外部因素，预测检测器 $i$ 在未来 $\tau$ 个小时的值，表示为 $\hat{\mathbf{y}}^i = (\hat{y}^i\_{T+1}, \hat{y}^i\_{T+2}, ..., \hat{y}^i\_{T+\tau})^{\rm T} \in \mathbb{R}^{\tau}$.

# 3 Multi-level Attention Networks

![Fig2](/images/geoman-multi-level-attention-networks-for-geo-sensory-time-series-prediction/Fig2.PNG)

图 2 展示了我们方法的框架。基于编码解码架构[Cho et al., 2014b]，我们用两个分开的 LSTM，一个对输入序列编码，也就是对历史的 geo-sensory 时间序列，另一个来预测输出的序列 $\hat{y}^i$。更具体的讲，我们的模型 GeoMAN 有两个主要部分：
1. 多级注意力机制。包含一个带有两类空间注意力机制的编码器和一个带有时间注意力的解码器。在编码器，我们开发了两种不同的注意力机制，局部空间注意力和全局空间注意力，如图 2 所示，这两种注意力机制通过前几步编码器的隐藏状态、前几步检测器的值和空间信息（检测器网络），可以在每个时间步上捕获检测器间的复杂关系。在解码器，我们使用了一个时间注意力机制来自适应地选择之前的时间段来做预测。
2. 外部因素融合。这个模块用来处理外部因素的影响，输出会作为解码器的一部分输入。这里，我们使用 $h\_t \in \mathbb{R}^m$ 和 $s\_t \in \mathbb{R}^m$ 来表示编码器在时间 $t$ 的隐藏状态和细胞状态。$d\_t \in \mathbb{R}^n$ 和 $s' \in \mathbb{R}^n$ 表示解码器的隐藏状态和细胞状态。

## 3.1 Spatial Attention

### Local Spatial Attention

我们先介绍空间局部注意力机制。对应一个监测器，在它的局部时间序列上有复杂的关联性。举个例子，一个空气质量检测站会记录不同的时间序列如 PM2.5，NO 和 SO2。事实上，PM2.5 的浓度通常被其他时间序列影响，包括其他的空气污染物和局部空气质量 [Wang et al., 2005]。为了解决这个问题，给定第 $i$ 个检测器第 $k$ 个局部特征向量 $\mathbf{x}^{i,k}$，我们使用注意力机制自适应地捕获目标序列和每个局部特征间的动态关系：

$$\tag{1}
e^k\_t = \mathbf{v}^T\_l \text{tanh} (\mathbf{W}\_l [\mathbf{h}\_{t-1};\mathbf{s}\_{t-1}] + \mathbf{U}\_l \mathbf{x}^{i,k} + \mathbf{b}\_l)
$$

$$\tag{2}
\alpha^k\_t = \frac{\text{exp}(e^k\_t)}{\sum^{N\_l}\_{j=1}\text{exp}(e^j\_t)}.
$$

其中 $[\cdot;\cdot]$ 是拼接操作（论文这里写的是 concentration，我怎么觉得是concatenation呢。。。）。$\mathbf{v}\_l, \mathbf{b}\_l \in \mathbb{R}^T, \mathbf{W}\_l \in \mathbb{R}^{T \times 2m}, \mathbf{U}\_l \in \mathbb{R}^{T \times T}$ 是参数。局部特征的注意力权重通过编码器中输入的局部特征和历史状态共同决定。这个注意力分数语义上表示每个局部特征的重要性，局部空间注意力在时间步 $t$ 的输出向量通过下式计算：

$$\tag{3}
\tilde{\mathbf{x}}^{local}\_t = (\alpha^1\_t x^{i,1}\_t, \alpha^2\_t x^{i,2}\_t, \dots, \alpha^{N\_l}\_t x^{i,N\_l}\_t)^{\rm T}.
$$

### Global Spatial Attention

对于一个监测器记录的目标时间序列，其他监测器是时间序列对其有直接影响。然而，影响权重是高度动态地，随时间变化。因为可能有很多不相关的序列，直接使用各种时间序列作为编码器的输入来捕获不同监测器之间的关系会导致很高的计算开销并且降低模型的能力。而且这样的影响权重受其他监测器的局部条件影响。举个例子，当强风从一个遥远地地方吹过来，这个区域的空气质量回比之前受影响的多。受这个现象的启发，我们开发了一个新的注意力机制捕获不同监测器间的动态关系。给定第 $i$ 个监测器作为我们的预测目标，另一个监测器 $l$，我们计算他们之间的注意力分数如下：

$$
g^l\_t = \mathbf{v}^{\rm T}\_g \text{tanh} (\mathbf{W}\_g [\mathbf{h}\_{t-1}; \mathbf{s}\_{t-1}] + \mathbf{U}\_g \mathbf{y}^l + \mathbf{W}'\_g \mathbf{X}^l \mathbf{u}\_g + \mathbf{b}\_g),
$$

其中 $\mathbf{v}\_g, \mathbf{u}\_g, \mathbf{b}\_g \in \mathbb{R}^T, \mathbf{W}\_g \in \mathbb{R}^{T \times 2m}, \mathbf{U}\_g \in \mathbb{R}^{T \times T}, \mathbf{W}'\_g \in \mathbb{R}^{T \times N\_l}$ 是参数。通过考虑目标序列和其他检测器的局部特征，这个注意力机制可以自适应地选择相关的监测器来做预测。同时，通过考虑编码器内前一隐藏状态和细胞状态，历史信息会跨时间流动。

注意，空间因素也会对不同监测器之间的关系做出贡献。一般来说，geo-sensors 通过一个明确的或隐含的网络连接起来。这里，我们使用一个矩阵 $\mathbf{P} \in \mathbb{R}^{N\_g \times N\_g}$ 来衡量地理空间相似度（如地理距离的倒数），$P\_{i,j}$ 表示监测器 $i$ 和 $j$ 之间的相似度。不同于注意力权重，地理相似度可以看作是先验知识。特别的说，如果 $N\_g$ 很大，一个方法是使用最近邻或最相近的一组而不是所有的监测器。之后，我们使用一个 softmax 函数，确定所有的注意力权重之和为1，两个方法共同考虑地理相似度如下：

$$\tag{4}
\beta^l\_t = \frac{\text{exp}((1-\lambda)g^l\_t + \lambda P\_{i,l})}{\sum^{N\_g}\_{j=1} \text{exp}((1-\lambda)g^j\_t + \lambda P\_{i,j})},
$$

其中，$\lambda$ 是一个可调的超惨。如果 $\lambda$ 大，这项会强制注意力权重等于地理相似度。全局注意力的输出向量计算如下：

$$\tag{5}
\tilde{\mathbf{x}}^{global}\_t = (\beta^1\_t y^1\_t, \beta^2\_t y^2\_t, \dots, \beta^{N\_g}\_t y^{N\_g}\_t)^{\rm T}.
$$

## 3.2 Temporal Attention

因为编码解码结构会随着长度增长会很快的降低性能，一个重要的扩展是增加时间注意力机制，可以自适应地选择编码器相关的隐藏状态来生成输出序列，即模型对目标序列中不同时间间隔的动态时间关系建模。特别来说，为了计算每个输出时间 $t'$ 对编码器每个隐藏状态的的注意力向量，我们定义：

$$\tag{6}
u^o\_{t'} = \mathbf{v}^{\rm T}\_d \text{tanh} (\mathbf{W}'\_d [\mathbf{d}\_{t'-1}; \mathbf{s}'\_{t'-1}] + \mathbf{W}\_d \mathbf{h}\_o + \mathbf{b}\_d),
$$

$$\tag{7}
\gamma^o\_{t'} = \frac{\text{exp} (u^o\_{t'})}{\sum^T\_{o=1} \gamma^o\_{t'} \mathbf{h}\_o},
$$

$$\tag{8}
\mathbf{c}\_{t'} = \sum^T\_{o=1} \gamma^o\_{t'} \mathbf{h}\_o,
$$

## 3.3 External Factor Fusion

Geo-sensory 时间序列和空间因素有强烈的相关性，如 POI 和监测器网络。这些因素一起表示一个区域的功能。而且还有很多时间因素影响监测器的数值，如气象或时间。受之前工作的启发 [Liang et al., 2017; Wang et al., 2018] 专注时空应用中的外部因素的影响，我们设计了一个简单有效的组件来处理这些因素。

如图 2 所示，我们先将包含时间、气象特征的时间因素和表示目标监测器的监测器ID融合。因为未来的天气条件未知，我们使用天气预报来提升性能。这些因素的大部分都是离散特征，不能直接放到神经网络里面，我们通过将离散特征分开放入不同的嵌入层，将离散特征转换为低维向量。根据空间因素，我们使用不同 POI 类型的密度作为特征。因为监测器的属性依赖实际情况，我们只使用网络的结构特征，如邻居数和交集等。最后，我们将获得的嵌入向量和空间特征拼接作为这个模块的输出，表示为 $\mathbf{ex}\_{t'} \in \mathbb{R}^{N\_e}$，其中 $t'$ 是解码器中未来的时间步。

## 3.4 Encoder-decoder & Model Training

编码器中，我们简单地从局部空间注意力和全局空间注意力聚合输出：

$$\tag{9}
\tilde{\mathbf{x}}\_t = [\tilde{\mathbf{x}}^{local}\_t; \tilde{\mathbf{x}}^{global}\_t],
$$

其中 $\tilde{\mathbf{x}}\_t \in \mathbb{R}^{N\_l + N\_g}$。我们将拼接 $\tilde{\mathbf{x}}\_t$ 作为编码器的新输入，使用 $\mathbf{h}\_t = f\_e(\mathbf{h}\_{t-1}, \tilde{\mathbf{x}}\_t)$ 更新时间 $t$ 的隐藏状态，$f\_e$ 是一个 LSTM 单元。

解码器中，一旦我们获得了时间 $t'$ 的上下文向量 $\mathbf{c}\_{t'}$ 的带权和，我们将他与外部因素融合模块的输出 $\mathbf{ex}\_{t'}$ 还有解码器的最后一个输出 $\hat{y}^i\_{t'-1}$ 融合，用 $\mathbf{d}\_{t'} = f\_d (\mathbf{d}\_{t'-1}, [\hat{y}^i\_{t'-1}; \mathbf{ex}\_{t'}; \mathbf{c}\_{t'}])$ 更新解码器的隐藏状态，$f\_d$ 是解码器中使用的 LSTM 单元。然后，我们讲上下文向量 $\mathbf{c}\_{t'}$ 和隐藏状态 $\mathbf{d}\_{t'}$ 拼接，得到新的隐藏状态，然后做最后的预测：

$$\tag{10}
\hat{y}^i\_{t'} = \mathbf{v}^{\rm T}\_y (\mathbf{W}\_m [\mathbf{c}\_{t'}; \mathbf{b}\_{t'}] + \mathbf{b}\_m) + b\_y,
$$

其中，$\mathbf{W}\_m \in \mathbb{R}^{n \times (m + n))}$ 和 $\mathbf{b}\_m \in \mathbb{R}^n$ 将 $[\mathbf{c}\_{t'}; \mathbf{d}\_{t'}] \in \mathbb{R}^{m + n}$ 映射到解码器隐藏状态的空间。最后，我们用一个线性变换生成最终结果。

因为我们的方法是平滑且可微的，可以通过反向传播训练。在训练时，我们使用 Adam，最小化 MSE。

$$\tag{11}
\mathcal{L}(\theta) = \Vert \hat{\mathbf{y}}^i - \mathbf{y}^i \Vert^2\_2,
$$

# 4 Experiments

## 4.1 Settings

### Datasets

我们在两个数据集中开展了实验，每个数据集包含三个子集：气象数据、POI、监测器网络数据。

- 水质数据集：中国东南的一个城市的供水系统中的监测器提供了长达三年的每5分钟一个记录的数据，包含了残余氯(RC)、浑浊度和PH值等。我们将 RC 作为目标时间序列，因为它在环境科学中作为常用的水质指标。一共有 14 个监测器，监测 10 个指标，它们之间通过管道网络相连。我们使用 Liu et al., 2016a 提出的指标作为这个数据集的相似度矩阵。

- 空气质量：从一个公开数据集抓取的，这个数据集包含不同污染物的浓度，还有气象数据，北京地区一共 35 个监测器。主要污染物是 PM2.5，因此我们将它作为目标。我们只使用空间距离的倒数表示两个监测器之间的相似度。

![Table1](/images/geoman-multi-level-attention-networks-for-geo-sensory-time-series-prediction/Table1.JPG)

对于水质数据集，我们将数据分成了不重叠的训练集、验证集和测试集，去年的前一半作为验证机，去年的后半段作为测试集。可惜的是，我们在第二个数据集上没能获得很多的数据，因此我们使用了8：1：1的比例划分。

### Evaluation Metrics

我们使用多个标准评价模型，RMSE 和 MAE。

### Hyperparameters

我们令 $\tau = 6$，做短期预测。在训练阶段，batch size 256，学习率 0.001。外部因素融合模块，我使用 $\mathbb{R}^6$ 嵌入监测器 ID，时间特征 $\mathbb{R}^10$。我们的模型一共 4 个超参数，$\lambda$ 根据经验设置，从 0.1 到 0.5。对于窗口长度 $T$，我们设为 $T \in \lbrace 6, 12, 24, 36, 48 \rbrace$。为了简单，我们将编码器和解码器采用同样的隐藏维数，网格搜索 $\lbrace 32, 64, 128, 256\rbrace$。我们堆叠 LSTM 来提高性能，层数记为 $q$。验证集上得到的最好参数是 $q = 2, m = n = 64, \lambda = 2$。

## 4.2 Baselines

ARIMA, VAR, GBRT, FFA, stMTMVL, stDNN, LSTM, Seq2seq, DA-RNN。

对于 ARIMA，我们用前六个小时的数据作为输入。stMTMVL 和 FFA，我们使用和作者一样的设置。和 GeoMAN 类似，我们使用前 $T \in \lbrace 6, 12, 24, 36, 48\rbrace$ 个小时的数据作为其他模型的输入。最后，我们测试了不同的超参数，得到了每个模型的最好效果。

## 4.3 Model Comparsion

我们在两个数据集上比较了模型和 baselines。为了公平，我们在表 2 展示了每个方法的最好性能。

我们的方法在水质预测上得到了最好的性能。比 state-of-the-art 的 DA-RNN 在两个指标上分别提升了 14.2% 和 13.5%。因为 RC 浓度有一个确定的周期模式，stDNN 和 基于 RNN 的模型比 stMTMVL 和 FFA 获得了更好的效果，因为他们能捕获更长的时间依赖。对比 LSTM 智能预测一个未来的时间步，GeoMAN 和 Seq2seq 由于解码器的存在有很大的提升。GBRT 比大部分方法也要好，体现了集成学习的优势。

对比数据相对稳定的水质数据集，PM2.5 的浓度有些时候震荡得很厉害，使得很难预测。表 2 展示了北京的空气质量数据集上一个全面的对比。可以看到我们的模型有最好的效果。我们主要讨论下 MAE。我们的方法比这些方法的 MAE 相对低 7.2% 和 63.5%，展示出了比其他方法更好的泛化效果。另一个有趣的现象是 stMTMVL 在水质预测上表现很好，在空气质量上