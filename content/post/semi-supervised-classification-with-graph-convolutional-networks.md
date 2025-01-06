---
categories:
- 论文阅读笔记
date: 2018-07-02 20:04:20+0000
draft: false
math: true
tags:
- deep learning
- machine learning
- graph convolutional network
- Graph
title: Semi-Supervised Classification With Graph Convolutional Networks
---
ICLR 2017。图卷积中谱图领域理论上很重要的一篇论文，提升了图卷积的性能，使用切比雪夫多项式的1阶近似完成了高效的图卷积架构。原文链接：[Semi-Supervised Classification with Graph Convolutional Networks. Kipf & Welling 2017](https://arxiv.org/abs/1609.02907v4)
<!--more-->


# 摘要
我们提出了一种在图结构数据上的半监督可扩展学习方法，基于高效的图卷积变体。契机是通过一个谱图卷积的局部一阶近似得到的我们的图卷积结构。我们的模型与图的边数呈线性关系，学习到的隐藏层可以对图的顶点和局部图结构同时进行编码。在引文网络和一个知识图谱数据集上的大量实验结果表明我们的方法比相关方法好很多。

# 引言
我们考虑一个对图顶点进行分类的问题，只有一小部分的顶点有标签。这个问题可以通过基于图的半监督学习任务建模，通过某些明确的图正则化方法(Zhu et al., 2003; Zhou et al., 2004; Belkin et al., 2006; Weston et al., 2012)可以平滑标签信息，举个例子，通过在loss function使用一个图拉普拉斯正则项：
$$\tag{1} \mathcal{L} = \mathcal{L\_0} + \lambda \mathcal{L\_{reg}}, \rm with \ \mathcal{L\_{reg}} = \sum\_{i.j}A\_{ij} \Vert f(X\_i) - f(X\_j) \Vert^2 = f(X)^T \Delta f(X)$$
其中，$\mathcal{L\_0}$表示对于图的标签部分的监督损失，$f(\cdot)$可以是一个神经网络类的可微分函数，$\lambda$是权重向量，$X$是定点特征向量$X\_i$的矩阵。$N$个顶点$v\_i \in \mathcal{V}$，边$(v\_i, v\_j) \in \varepsilon$，邻接矩阵$A \in \mathbb{R}^{N \times N}$（二值的或者带权重的），还有一个度矩阵$D\_{ii} = \sum\_jA\_{ij}$。式1依赖于“图中相连的顶点更有可能具有相同的标记”这一假设。然而，这个假设，可能会限制模型的能力，因为图的边并不是必须要编码成相似的，而是要包含更多的信息。
在我们的研究中，我们将图结构直接通过一个神经网络模型$f(X, A)$进行编码，并且在监督的目标$\mathcal{L\_0}$下对所有有标记的顶点进行训练，因此避免了损失函数中刻意的对图进行正则化。在图的邻接矩阵上使用$f(\cdot)$可以使模型从监督损失$\mathcal{L\_0}$中分布梯度信息，并且能够从有标记和没有标记的顶点上学习到他们的表示。
我们的贡献有两点，首先，我们引入了一个简单的，表现很好的针对神经网络的对层传播规则，其中，这个神经网络是直接应用到图上的，并且展示了这个规则是如何通过谱图卷积的一阶近似启发得到的。其次，我们展示了这种形式的基于图的神经网络可以用于对图中的顶点进行更快更可扩展的半监督分类任务。在大量数据集上的实验表明我们的模型在分类精度和效率上比当前在半监督学习中的先进算法要好。

# 图上的快速近似卷积
在这部分，我们会讨论一个特殊的基于图的神经网络$f(X, A)$。考虑一个多层图卷积网络(GCN)，通过以下的传播规则：
$$\tag{2} H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)})$$
其中，$\tilde{A} = A + I\_N$是无向图$\mathcal{G}$加了自连接的邻接矩阵。$I\_N$是单位阵，$\tilde{D}\_{ii} = \sum\_j \tilde{A}\_{ij}$，$W^{(l)}$是一个针对层训练的权重矩阵。$\sigma(\cdot)$表示一个激活函数，比如$\rm ReLU(\cdot) = \rm max(0, \cdot)$，$H^{(l)} \in \mathbb{R}^{N \times D}$是第$l$层的激活矩阵；$H^{(0)} = X$。接下来我们将展示通过图上的一阶近似局部谱滤波器(Hammond et al., 2011; Defferrard et al., 2016)的传播过程。

## 谱图卷积 spectral graph convolutions
定义图上的谱图卷积为信号$x \in \mathbb{R}^N$和一个滤波器$g\_\theta = \rm diag(\theta)$，参数是傅里叶域中的$\theta \in \mathbb{R}^N$，也就是：
$$\tag{3} g\_\theta \ast x = U g\_\theta U^T x$$
其中$U$是归一化的拉普拉斯矩阵$L = I\_N - D^{-\frac{1}{2}}AD^{-\frac{1}{2}} = U \Lambda U^T$的特征向量组成的矩阵，$\Lambda$是特征值组成的对角阵，$U^Tx$是$x$的图傅里叶变换。可以认为$g\_\theta$是关于$L$的特征值的函数，也就是说$g\_\theta(\Lambda)$。式3的计算量很大，因为特征向量矩阵$U$的乘法的时间复杂度是$O(N^2)$。此外，对于大尺度的图来说，对$L$进行特征值分解是计算量非常大的一件事。为了避开这个问题，Hammond et al.(2001)建议使用$K$阶切比雪夫多项式$T\_k(x)$来近似$g\_\theta(\Lambda)$：
$$\tag{4} g\_\theta' \approx \sum^K\_{k=0} \theta'\_k T\_k(\tilde{\Lambda})$$
其中，$\tilde{\Lambda} = \frac{2}{\lambda\_{max}} \Lambda - I\_N$。$\lambda\_{max}$表示$L$的最大特征值。$\theta' \in \mathbb{R}^K$是切比雪夫系数向量。切比雪夫多项式的定义是：$T\_k(x) = 2xT\_{k-1}(x) - T\_{k-2}(x)$，$T\_0(x) = 1$，$T\_1(x) = x$。
回到我们对于一个信号$x$和一个滤波器$g\_\theta'$的卷积的定义：
$$\tag{5} g\_\theta' \ast x \approx \sum^K\_{k=0} \theta'\_k T\_k(\tilde{L}) x$$
其中，$\tilde{L} = \frac{2}{\lambda x\_{max}} L - I\_N$；注意$(U \Lambda U^T)^k = U \Lambda^k U^T$。这个表达式目前是$K$阶局部的，因为这个表达式是拉普拉斯矩阵的$K$阶多项式，也就是说从中心节点向外最多走$K$步，$K$阶邻居。式5的时间复杂度是$O(\vert \varepsilon \vert)$，也就是和边数呈线性关系。Defferrard et al.(2016)使用这个$K$阶局部卷积定义了在图上的卷积神经网络。

## 按层的线性模型 layer-wise linear model
一个基于图卷积的神经网络模型可以通过堆叠式5这样的多个卷积层来实现，每层后面加一个非线性激活即可。现在假设$K=1$，也就是对$L$线性的一个函数，因此得到一个在图拉普拉斯谱(graph Laplacian spectrum)上的线性函数。这样，我们仍然能通过堆叠多个这样的层获得一个卷积函数，但是我们就不会再受限于明显的参数限制，比如切比雪夫多项式。我们直觉上期望这样一个模型可以减轻在度分布很广泛的图上局部图结构模型过拟合的问题，如社交网络、引文网络、知识图谱和其他很多真实数据集。此外，这个公式可以让我们搭建更深的网络，一个可以提升模型学习能力的实例是He et al., 2016。
在GCN的线性公式中，我们让$\lambda\_{max}$近似等于2，因为我们期望神经网络参数可以在训练中适应这个变化。在这个近似下，式5可以简化为：
$$\tag{6} g\_\theta' \ast x \approx \theta'\_0x + \theta'\_1 (L - I\_N)x = \theta'\_0x - \theta'\_1 D^{-\frac{1}{2}} A D^{-\frac{1}{2}} x$$
两个参数$\theta'\_0$和$\theta'\_1$。滤波器参数可以在整个图上共享。连续的使用这种形式的卷积可以有效的对一个顶点的$k$阶邻居进行卷积，$k$是连续的卷积操作或模型中卷积层的个数。
实际上，通过限制参数的数量可以进一步的解决过拟合的问题，并且最小化每层的操作数量（比如矩阵乘法）。这时的我们得到了下面的式子：
$$\tag{7} g\_\theta \ast x \approx \theta(I\_N + D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) x$$
只有一个参数$\theta = \theta'\_0 = - \theta'\_1$。注意，$I\_N + D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$现在的特征值在$[0, 2]$之间。在深层模型中重复应用这个操作会导致数值不稳定和梯度爆炸、消失的现象。为了减轻这个问题，我们引入了如下的重新正则化技巧：$I\_N + D^{-\frac{1}{2}} A D^{-\frac{1}{2}} \to \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$，$\tilde{A} = A + I\_N$，$\tilde{D}\_{ii} = \sum\_j \tilde{A}\_{ij}$。
我们可以将这个定义泛化到一个有着$C$个通道的信号$X \in \mathbb{R}^{N \times C}$上，也就是每个顶点都有一个$C$维的特征向量，对于$F$个滤波器或$F$个feature map的卷积如下：
$$\tag{8} Z = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X \Theta$$
其中$\Theta \in \mathbb{R}^{C \times F}$是一个滤波器的参数矩阵，$Z \in \mathbb{R}^{N \times F}$是卷积的信号矩阵。卷积操作的时间复杂度是$O(\vert \varepsilon \vert F C)$，因为$\tilde{A} X$可以被实现成一个稀疏矩阵和一个稠密矩阵的乘积。

# 半监督顶点分类
介绍过这个简单、灵活的可以在图上传播信息的模型$f(X, A)$后，我们回到半监督顶点分类的问题上。如介绍里面所说的，我们可以减轻在基于图的半监督学习任务中的假设，通过在图结构上的数据$X$和邻接矩阵$A$上使用模型$f(X, A)$。我们期望这个设置可以在邻接矩阵表达出数据$X$没有的信息的这种情况时表现的很好，比如引文网络中，引用的关系或是知识图谱中的关系。整个模型是一个多层的GCN，如图1所示。
![Fig1](/images/semi-supervised-classification-with-graph-convolutional-networks/Fig1.PNG)

## 例子
我们考虑一个两层GCN对图中的顶点进行半监督分类，邻接矩阵是对称的。我们首先在预处理中计算$\hat{A} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$。前向传播模型的形式如下：
$$\tag{9} Z = f(X, A) = \rm softmax( \hat{A} \ \rm ReLU( \hat{A}XW^{(0)})W^{(1)})$$
这里，$W^{(0)} \in \mathbb{R}^{C \times H}$是输入到隐藏层的权重矩阵，有$H$个feature map。$W^{(1)} \in \mathbb{R}^{H \times F}$是隐藏层到输出的权重矩阵。softmax激活函数定义为$\rm softmax(x\_i) = \frac{1}{\mathcal{Z}} \exp(x\_i)$，$\mathcal{Z} = \sum\_i \exp(x\_i)$，按行使用。对于半监督多类别分类，我们使用交叉熵来衡量所有标记样本的误差：
$$\tag{10} \mathcal{L} = - \sum\_{l \in \mathcal{Y}\_L} \sum^F\_{f = 1} Y\_{lf} \ln(Z\_{lf})$$
其中，$\mathcal{Y}\_L$是有标签的顶点的下标集合。
神经网络权重$W^{(0)}$和$W^{(1)}$使用梯度下降训练。我们每次训练的时候都是用全部的训练集来做梯度下降，只要数据集能放到内存中。对$A$进行稀疏矩阵的表示，内存的使用量是$O(\vert \varepsilon \vert)$。训练过程中使用了dropout增加随机性。我们将在未来的工作使用mini-batch随机梯度下降。

## 实现
我们使用Tensorflow实现了基于GPU的，稀疏稠密矩阵乘法形式。式9的时间复杂度是$O(\vert \varepsilon \vert C H F)$。