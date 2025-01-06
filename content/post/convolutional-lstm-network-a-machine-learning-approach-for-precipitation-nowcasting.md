---
categories:
- 论文阅读笔记
date: 2018-09-27 15:11:55+0000
draft: false
math: true
tags:
- deep learning
- Spatial-temporal
title: 'Convolutional LSTM Network: A Machine Learning Approach for Precipitation
  Nowcasting'
---
NIPS 2015. 将 FC-LSTM 中的全连接换成了卷积，也就是将普通的权重与矩阵相乘，换成了卷积核对输入和隐藏状态的卷积，为了能捕获空间信息，将输入变成了4维的矩阵，后两维表示空间信息。两个数据集：Moving-MNIST 和 雷达云图数据集。原文链接：[Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)

<!--more-->

# Abstract

降雨量预测的目标是预测未来一个局部区域再相对短的一个时间段的降雨密度。之前几乎没有研究从机器学习角度研究这个重要而且很有挑战的问题。在这篇论文中，我们将降雨量预测问题定义成一个时空序列预测问题，输入和预测都是时空序列。通过扩展 *fully connected LSTM* (FC-LSTM)，在输入到隐藏与隐藏到隐藏的变换都加入卷积结构，我们提出了 *convolutional LSTM* (ConvLSTM)，用它做了一个端到端的模型来预测降雨量。实验表明我们的 ConvLSTM 网络比 FC-LSTM 以及最先进的 ROVER 算法在降水量预测上能更好地捕获时空关系。

# 1. Introduction

# 2. Preliminaries

## 2.1 Forumulation of Precipitation Nowcasting Problem

降水量预测的目标是使用之前的雷达声波序列预测一个区域（如香港、纽约、东京）未来定长（时间）的雷达图。在实际应用中，雷达图通常从气候雷达每6到10分钟获得一次，然后预测未来1到6小时，也就是说，预测6到60帧。从机器学习的角度来看，这个问题可以看作是时空序列预测问题。

假设我们在一个空间区域观测了一个动态系统，由一个 $M \times N$ 的网格组成，$M$ 行 $N$ 列。在网格的每个单元格内部随着时间的变化，有 $P$ 个测量值。因此，观测值在任意时刻可以表示成一个张量 $\mathcal{X} \in \mathbf{R}^{P \times M \times N}$，其中 $\mathbf{R}$ 表示观测到的特征的定义域。如果我们周期性的记录观测值，我们可以得到一个序列 $\hat{\mathcal{X}\_1}, \hat{\mathcal{X}\_2}, ..., \hat{\mathcal{X}\_t}$。这个时空序列预测问题是给定前 $J$ 个观测值，预测未来长度为 $K$ 的序列：

$$\tag{1}
\tilde{\mathcal{X}}\_{t+1}, ..., \tilde{\mathcal{X}}\_{t+K} = \mathop{\arg \max}\_{\mathcal{X}\_{t+1}, ...\mathcal{X}\_{t+K}} p(\mathcal{X}\_{t+1}, ..., \mathcal{X}\_{t+K} \mid \hat{\mathcal{X}}\_{t-J+1}, \hat{\mathcal{X}}\_{t-J+2}, ..., \hat{\mathcal{X}}\_t)
$$

对于降雨量预测，每个时间戳的观测值是一个2D雷达地图。如果我们将地图分到平铺且不重合的部分，将每个部分内的像素看作是它的观测值（图1），预测问题就会自然地变成一个时空序列预测问题。

![Figure1](/images/convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting/Fig1.JPG)

我们注意到我们的时空序列预测问题与一步时间序列预测问题不同，因为我们问题的预测目标是一个包含时间和空间结构的序列。尽管长度为$K$的序列中的自由变量的数量可以达到$O(M^KN^KP^K)$，实际上我们可以挖掘可能的预测值的空间结构，减小维度，使问题变得容易处理。

## 2.2 Long Short-Term Memory for Sequence Modeling

这篇论文，我们使用 FC-LSTM 的公式[11]

$$\tag{2}
\begin{aligned}
i\_t &= \sigma ( W\_{xi} x\_t + W\_{hi} h\_{t-1} + W\_{ci} \circ c\_{t-1} + b\_i) \\
f\_t &= \sigma ( W\_{xf} x\_t + W\_{hf} h\_{t-1} + W\_{cf} \circ c\_{t-1} + b\_f) \\
c\_t &= f\_t \circ c\_{t-1} + i\_t \circ \mathrm{tanh}(W\_{xc} x\_t + W\_{hc} h\_{t-1} + b\_c) \\
o\_t &= \sigma ( W\_{xo} x\_t + W\_{ho} h\_{t-1} + W\_{co} \circ c\_t + b\_o ) \\
h\_t &= o\_t \circ \mathrm{tanh}(c\_t)
\end{aligned}
$$

多个 LSTM 可以堆叠，对复杂的结构在时间上拼接。

# 3 The Model

尽管 FC-LSTM 层已经在时间关系上表现的很有效了，但是在空间数据上有很多的冗余。为了解决这个问题，我们提出了 FC-LSTM 的扩展，在输入到隐藏，以及隐藏到隐藏的变换上有了卷积的结构。通过堆叠多个 ConvLSTM 层，形成一个 encoding-forecasting 结构，我们可以构建一个不仅可以处理降雨量预测问题，还可以处理更一般的时空序列预测问题的模型。

## 3.1 Convolutional LSTM

FC-LSTM 的主要问题是处理时空数据的时候，它的全连接层没有对空间信息进行编码。为了解决这个问题，我们的设计中的一个特征就是，所有的输入 $\mathcal{X}\_1, ..., \mathcal{X}\_t$，细胞状态 $\mathcal{C}\_1, ..., \mathcal{C}\_t$，隐藏状态 $\mathcal{H}\_1, ... \mathcal{H}\_t$，还有门 $i\_t, f\_t, o\_t$ 都是三维的张量，而且后两维都是空间维（行和列）。为了更好的理解输入和状态，我们可以把它们想象成在空间网格站立的向量。ConvLSTM 通过输入和局部邻居的上一个状态决定了一个特定细胞的未来状态。通过在输入到隐藏，隐藏到隐藏中使用一个卷积操作器就可以轻松的实现（图2）。ConvLSTM的重要的公式如（3）所示（下面的公式），$\ast$表示卷积操作，$\circ$表示Hadamard积：

$$\tag{3}
\begin{aligned}
i\_t &= \sigma( W\_{xi} \ast \mathcal{X}\_t + W\_{hi} \ast \mathcal{H}\_{t-1} + W\_{ci} \circ \mathcal{C}\_{t-1} + b\_i )\\
f\_t &= \sigma( W\_{xf} \ast \mathcal{X}\_t + W\_{hf} \ast \mathcal{H}\_{t-1} + W\_{cf} \circ \mathcal{C}\_{t-1} + b\_f )\\
\mathcal{C}\_t &= f\_t \circ \mathcal{C}\_{t-1} + i\_t \circ \mathrm{tanh}(W\_{xc} \ast \mathcal{X}\_t + W\_{hc} \ast \mathcal{H}\_{t-1} + b\_c)\\
o\_t &= \sigma( W\_{xo} \ast \mathcal{X}\_t + W\_{ho} \ast \mathcal{H}\_{t-1} + W\_{co} \circ \mathcal{C}\_t + b\_o )\\
\mathcal{H}\_t &= o\_t \circ \mathrm{tanh}(\mathcal{C}\_t)
\end{aligned}
$$

![Figure2](/images/convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting/Fig2.JPG)

如果我们将状态看作是移动物体的隐藏表示，带有一个更大的变换卷积核的 ConvLSTM 应该能捕获更快的移动，而小的核能捕获慢的移动。同时，如果我们用[16]的角度来看，由式2表示的传统的 FC-LSTM 的输入、细胞输出以及隐藏状态可以看作是一个后两维都是1的三维张量。这样的话，FC-LSTM实际上是所有特征都站立在一个单独细胞上 ConvLSTM 的一个特例。

为了确保状态和输入有相同个数的行和列，需要在卷积操作之前加入 padding 操作。这里隐藏状态在边界点的 padding 可以看作是使用 *state of the outside world* 来计算。通常，在第一个输入来之前，我们将 LSTM 的所有状态初始化为0，对应未来的 “total ignorance”。相似地，如果我们在隐藏状态上用 zero-padding，我们实际上将 *state of the outside world* 设定为0，而且假设没有关于外部的先验知识。通过在状态上加 padding，我们可以不同地对待边界点，很多时候这都是有用的。举个例子，假设我们的系统正在观测一个被墙环绕的移动的球。尽管我们不能看到这些墙，但是我们可以通过观察球的一次次反弹推测它们的存在，如果边界点像内部的点一样有相同的状态变化动态性，那这就几乎不可能了。

## 3.2 Encoding-Forecasting Structure

就像 FC-LSTM，ConvLSTM 可以使用块对复杂的结构建模。对于我们的时空序列预测问题，我们使用图3这样的结构，由两个网络组成，一个编码网络，一个预测网络。就像[21]，预测网络的初始状态和细胞输出从编码网络的最后一个状态复制过来。两个网络都通过堆叠多个 ConvLSTM 层构成。因为我们的预测目标与输入的维度相同，我们将预测网络所有的状态拼接，放到一个 $1 \times 1$ 的卷积层中，生成最后的预测结果。

![Figure3](/images/convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting/Fig3.JPG)

我们使用像[23]一样的观点解释这个结构。编码 LSTM 将整个输入序列压缩到一个隐藏状态的张量中，预测 LSTM 解压了这个状态，给出了最后的预测：

$$\tag{4}
\begin{aligned}
\tilde{\mathcal{X}\_{t+1}}, ..., \tilde{\mathcal{X}\_{t+K}} &= \mathop{\arg \max}\_{\mathcal{X}\_{t+1}, ..., \mathcal{X}\_{t+K}} p(\mathcal{X}\_{t+1}, ..., \mathcal{X}\_{t+K} \mid \hat{\mathcal{X}}\_{t-J+1}, \hat{\mathcal{X}}\_{t-J+2}, ..., \hat{\mathcal{X}}\_t) \\
&\approx \mathop{\arg \max}\_{\mathcal{X}\_{t+1}, ..., \mathcal{X}\_{t+K}} p(\mathcal{X}\_{t+1}, ..., \mathcal{X}\_{t+K} \mid f\_{encoding} (\hat{\mathcal{X}}\_{t-J+1}, \hat{\mathcal{X}}\_{t-J+2}, ..., \hat{\mathcal{X}}\_t)) \\
&\approx g\_{forecasting}(f\_{encoding}(\hat{\mathcal{X}}\_{t-J+1}, \hat{\mathcal{X}}\_{t-J+2}, ..., \hat{\mathcal{X}}\_t))
\end{aligned}
$$

这个结构与[21]中的 LSTM 未来预测模型相似，除了我们的输入和输出元素都是三维的张量，保留了所有的空间信息。因为网络堆叠了多个 ConvLSTM 层，它由很强的表示能力可以在复杂的动态系统中给出预测，如降雨量预测问题。

# 4. Experiments

我们将我们的模型 ConvLSTM 与 FC-LSTM 在一个人工生成的 Moving-MNIST 数据集上做了对比，对我们的模型进行一个初步的了解。我们使用了不同的层数以及不同的卷积核大小，也研究了一些 "out-of-domain" 的情况，如[21]。为了验证我们的模型在更有挑战的降雨量预测问题上的有效性，我们构建了一个新的雷达声波图，在几个降雨量预测的指标上，比较了我们的模型和当前最先进的 ROVER 算法。结果显示这两个数据集上：

1. ConvLSTM 比 FC-LSTM 在处理时空关系时更好
2. 隐藏状态到隐藏状态的卷积核的尺寸大于1对于捕获时空运动模式来说很重要
3. 深、且参数少的模型能生成更好的结果
4. ConvLSTM 比 ROVER 在降雨量预测上表现的更好。