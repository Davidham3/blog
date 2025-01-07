---
categories:
- 论文阅读笔记
date: 2019-06-25 16:34:02+0000
description: ICML 2019，原文地址：[Self-Attention Graph Pooling](https://arxiv.org/abs/1904.08082)
draft: false
math: true
tags:
- deep learning
- Graph
- graph convolutional network
title: Self-Attention Graph Pooling
---
ICML 2019，原文地址：[Self-Attention Graph Pooling](https://arxiv.org/abs/1904.08082)
<!--more-->

# Abstract

这些年有一些先进的方法将深度学习应用到了图数据上。研究专注于将卷积神经网络推广到图数据上，包括重新定义图上的卷积和下采样（池化）。推广的卷积方法已经被证明有性能提升且被广泛使用。但是，下采样的方法仍然是一个难题且有提升空间。我们提出了一个基于自注意力的图的池化方法。使用图卷积的自注意力使得我们的池化方法可以同时考虑顶点特征和图的拓扑结构。为了确保一个公平的对比，我们使用了相同的训练步骤和模型架构。实验结果显示我们的方法有更高的分类精度。

# 1. Introduction

CNN 成功利用了图像、语音、视频数据中的欧氏空间（网格结构）。CNN 由卷积层和下采样层（池化层）组成。卷积层和池化层挖掘了网格数据的平移不变性和compositionality（这个我不知道是什么。。。）。结果是，CNN 用少量的参数就可以表现的很好。

然而，很多数据是非欧空间上的。社交网络、生物蛋白质网络、分子网络可以表示成网络。将 CNN 应用在非欧空间上的尝试已经获得了成功。很多研究重新定义了图上的卷积和池化。

对于图卷积的池化操作现在比较少。之前图的池化的研究只考虑图的拓扑结构 (Defferrard et al., 2016; Rhee et al., 2018)。一些方法利用了结点的特征获得一个小图的表示。最近，Ying et al.; Gao & Ji; Cangea et al. 提出了创新的池化方法，可以层级的表示图。这些方法使得图神经网络可以通过端到端的形式，在池化后获得尺寸缩减的图。

然而，上述的池化方法仍有提升空间。举个例子，Ying et al. 的可微层级池化方法有平方级别的空间复杂度，参数依赖于顶点数。Gao & Ji; Cangea et al. 解决了复杂度的问题，但是没有考虑图的拓扑结构。

我们提出的 SAGPool 是一个层次的自注意力图池化方法。我们的方法可以通过端到端的方式使用相对较少的参数学习到层次表示。自注意力机制用来区分结点是否丢弃掉还是保留。由于自注意力机制使用图卷积计算注意力分数，结点特征和图的拓扑结构可以被考虑其中。一句话，SAGPool 有前面方法的优点，是第一个使用自注意力用于池化的方法，并且获得了很好的性能。代码已经在 Github 上开源了。

# 2. Related Work

## 2.1. Graph Convolution

图上的卷积要么是基于谱的，要么是非谱的。谱方法专注于在傅里叶域上定义卷积，利用使用图拉普拉斯矩阵的谱滤波器。Kipf & Welling 提出了一个层级传播的规则，简化了使用切比雪夫展开来趋近拉普拉斯矩阵的方法。非谱方法的目标是定义一个卷积操作，可以直接应用在图上。通常来说，非谱方法，中心结点在特征传入下层之前聚合邻接结点的特征。Hamilton et al. 提出了 GraphSAGE，通过采样和聚合学习结点的嵌入。尽管 GraphSAGE 会采样固定数量的邻居，GAT 基于注意力机制，在所有的邻居上计算结点表示。两个方法在图相关的任务上都有提升。

## 2.2. Graph Pooling

池化层通过缩减表示的大小，使得 CNN 能减少参数的数量，因此能避免过拟合。为了泛化 CNN，GNN 上的池化是必要的。图的池化方法可以归入三类：基于拓扑的，基于全局的，基于层次的。

**Topology based pooling** 早期的工作使用图的缩减算法，而不是神经网络。谱聚类算法使用特征值分解获得缩减的图。然而，特征值分解的时间复杂度高。Graclus (Dhillon et al., 2007) 不使用特征向量计算给定图的聚类结果，而是通过一个谱聚类的目标函数与一个带权的核 k-means 目标函数的等价性。即便在最近的 GNN 模型中，Graclus 也被使用作为一个池化单元。

**Global pooling** 不像之前的方法，全局池化方法考虑图的特征。全局池化方法在每层聚合表示的时候使用加和的方式而不是用神经网络。这个方法可以处理不同结构的图，因为它获得了所有的表示。Gilmer et al. 将 GNN 看作是一种信息的传递规则，提出了一个通用框架用于图分类，整个图的表示可以通过使用 Set2Set (Vinyals et al., 2015) 来获得。SortPool (Zhang et al., 2018b) 根据一个图的结构角色对结点嵌入排序，将排序后的表示传入下一层。

**Hierarchical pooling** 全局池化方法不学习层次表示，但是对于捕获图的结构信息来说，层次表示很关键。层次池化的动机在于在每层构建一个模型，这个模型可以学习基于特征的或基于拓扑的顶点分配。Ying et al. 提出了 DiffPool，这是一种可微的图的池化方法，可以以端到端的形式学习分配矩阵。在层 $l$ 学习到的分配矩阵 $S^{(l)} \in \mathbb{R}^{n\_l \times n\_{l+1}}$ 包含了层 $l$ 中的结点在 $l + 1$ 层被分配到类簇的概率。$n\_l$ 表示层 $l$ 的结点数。结点通过下式来分配：

$$\tag{1}
S^{(l)} = \text{softmax}(\text{GNN}\_l (A^{(l)}, X^{(l)})) \\
A^{(l+1)} = S^{(l)\text{T}} A^{(l)} S^{(l)}
$$

$X$ 表示矩阵的结点特征，$A$ 是邻接矩阵。

Cangea et al. 使用 gPool (Gao & Ji, 2019) 获得了和 DiffPool 相当的性能。gPool 需要 $\mathcal{O}(\vert V \vert + \vert E \vert)$ 的空间复杂度，DiffPool 需要 $\mathcal{O}(k \vert V \vert^2)$ 的空间复杂度。$V$，
$E$，$k$ 分别表示顶点、边、池化比例。gPool 使用一个可学习的向量 $p$ 计算投影分数，然后使用这个分数选择最高的结点。投影分数通过 $p$ 和所有结点的特征向量的内积获得。分数表示结点可以获得的信息量。下面的式子大体的描述了 gPool 中的池化步骤：

$$\tag{2}
y = X^{(l)} \mathbf{p}^{(l)} / \Vert \mathbf{p}^{(l)} \Vert, \text{idx=top-rank}(y, \lceil kN \rceil) \\
A^{(l+1)} = A^{(l)}\_{\text{idx,idx}}
$$

如式 2，图的拓扑结构不影响投影分数。

为了进一步提高图的池化，我们提出了 SAGPool，可以在可观的时间和空间复杂度上利用特征和拓扑结构生成层次表示。

# 3. Proposed Method

SAGPool 的关键是它使用了 GNN 得到的注意力分数。SAGPool 层和模型架构分别是图 1 和图 2.

![Figure1](/blog/images/self-attention-graph-pooling/Fig1.JPG)

![Figure2](/blog/images/self-attention-graph-pooling/Fig2.JPG)

## 3.1 Self-Attention Graph Pooling

**Self-attention mask** 注意力机制广泛应用在最近的深度学习研究中。这样的机制使得模型可以更专注于重要的特征，不那么关注不重要的特征。自注意力一般称为内注意力，允许输入特征作为自身注意力的标准 (Vaswani et al., 2017)。我们使用图卷积获得自注意力分数。举个例子，如果图卷积的公式是 Kipf & Welling 使用的，那么自注意力分数 $Z \in \mathbb{R}^{N \times 1}$ 通过下式计算：

$$\tag{3}
Z = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X \Theta\_{att})
$$

$\sigma$ 是激活函数，如 $tanh$，$\tilde{A} \in \mathbb{R}^{N \times N}$ 是有自连接的邻接矩阵，$\tilde{D} \in \mathbb{R}^{N \times N}$ 是度矩阵，$X \in \mathbb{R}^{N \times F}$ 是图的特征矩阵，$\Theta\_{att} \in \mathbb{R}^{F \times 1}$ 是 SAGPool 层仅有的参数。通过利用图卷积获得自注意力分数，池化的结果是同时基于图的特征和拓扑结构的。我们利用 Gao & Ji; Cangea et al. 的结点选择方法，保留了输入的图的一部分结点，甚至当图的尺寸和结构改变时。池化比例 $k \in (0, 1]$ 是一个超参数决定了保留多少结点。基于 $Z$ 的值选择最高的 $\lceil kN \rceil$ 个结点。

$$\tag{4}
\text{idx = top-rank}(Z, \lceil kN \rceil), Z\_{mask} = Z\_{\text{idx}}
$$

$\text{top-rank}$ 返回最高的 $\lceil kN \rceil$ 个值的下标，$\cdot\_{\text{idx}}$ 是下标操作，$Z\_{mask}$ 是特征的注意力 mask。

**Graph pooling** 输入的图通过图 1 中的 **masking** 操作。

$$\tag{5}
X' = X\_{idx,:}, X\_{out} = X' \odot Z\_{mask}, A\_{out} = A\_{\text{idx, idx}}
$$

其中 $X\_{\text{idx,:}}$ 是指定行下标的特征矩阵，每行表示一个结点，$\odot$ 是 elementwise 乘积，$A\_{\text{idx, idx}}$ 是指定行下标和列下标的邻接矩阵。$X\_{out}$ 和 $A\_{out}$ 是新的特征矩阵和对应的邻接矩阵。

**Variation of SAGPool** 使用图卷积的主要原因是为了反映图的特征和拓扑结构。可以使用不同的图卷积来替换式 3 中的图卷积。计算注意力机制 $Z \in \mathbb{R}^{N \times 1}$ 的泛化公式如下：

$$\tag{6}
Z = \sigma(\text{GNN}(X, A))
$$

$X$ 和 $A$ 是特征矩阵和邻接矩阵。

除了使用邻接结点还可以使用多跳结点来计算注意力分数。式 7 和式 8 分别使用了两跳连接和堆叠 GNN 层。增加邻接矩阵的平方增加了两条邻居：

$$\tag{7}
Z = \sigma(\text{GNN}(X, A + A^2))
$$

堆叠 GNN 层可以间接的聚合两跳结点。这样的话，非线性层和参数的数量就增加了：

$$\tag{8}
Z = \sigma(\text{GNN}\_2 (\sigma(\text{GNN}\_1 (X, A)), A))
$$

式 7 和式 8 可以利用多跳连接。

另一个变体是平均多个注意力分数。平均注意力分数通过 $M$ 个 GNN 获得：

$$\tag{9}
Z = \frac{1}{m} \sum\_m \sigma(\text{GNN}\_m (X, A))
$$

在论文中，式 7，8，9 的模型分别记为 $\rm {SAGPool}\_{augmentation}$，$\rm {SAGPool}\_{serial}$，$\rm {SAGPool}\_{parallel}$。

## 3.2 Model Architecture

根据 Lipton & Steinhardt 的研究，如果对一个模型做很多修改，那很难知道是哪部分改进起的作用。为了一个公平的对比，我们使用了 Zhang et al. 和 Cangea et al. 的模型来对比我们的方法。

**Convolution layer** 如 2.1 节提到的，有很多图卷积的定义。其他类型的图卷积可能也能提升性能，但是我们利用的是 Kipf & Welling 提出的广泛使用的图卷积。式 10 和式3 一样，除了 $\Theta$ 的维度：

$$\tag{10}
h^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} h^{(l)} \Theta)
$$

其中 $h^{(l)}$ 是第 $l$ 层的节点表示，$\Theta \in \mathbb{R}^{F \times F'}$ 是卷积核。使用 ReLU 作为激活函数。

**Readout layer** 受 JK-net 的启发，Cangea et al. 提出了一个 readout 层，聚合结点的特征生成一个固定大小的表示。readout 层的聚合特征如下：

$$\tag{11}
s = \frac{1}{N} \sum^N\_{i=1} x\_i \mid \mid \mathop{max}\limits^N\_{i=1} x\_i
$$

$N$ 是结点数，$x\_i$ 是第 $i$ 个结点的特征向量，$\mid \mid$ 表示拼接。

**Global pooling architecture** 我们实现了 Zhang et al. 提出的全局池化结构。如图 2 所示，全局池化结构由三层图卷积层组成，每层的输出拼接在一起。结点特征在 readout 层聚合，然后接一个池化层。图的特征表示传入线性层用来分类。

**Hierarchical pooling architecture** 在这部分设置中，我们实现了 Cangea et al. 的层次池化结构。如图 2 所示，结构包含了三个块，每个块由一个卷积层和一个池化层组成。每个块的输出通过一个 readout 层聚合。每个 readout 层的输出之和放入线性层做分类。

![Table1](/blog/images/self-attention-graph-pooling/Table1.JPG)

# 4. Experiments

我们在图分类上评估了全局池化和层次池化。

## 4.1. Datasets

5 个数据集。

![Table2](/blog/images/self-attention-graph-pooling/Table2.JPG)

![Table3](/blog/images/self-attention-graph-pooling/Table3.JPG)

![Table4](/blog/images/self-attention-graph-pooling/Table4.JPG)

![Figure3](/blog/images/self-attention-graph-pooling/Fig3.JPG)