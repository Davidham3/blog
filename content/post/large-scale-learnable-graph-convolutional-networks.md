---
categories:
- 论文阅读笔记
date: 2018-09-17 15:22:43+0000
description: KDD 2018.将图结构数据变换到网格状数据中，使用传统的一维卷积进行卷积。变换的方式是：针对每个特征的大小，对邻居结点进行排序，取这个特征前k大的数作为它邻居这列特征的k个值。如果邻居不够，那就用0来补。这样就能得到该顶点的邻居信息，组成一个矩阵，然后使用一维卷积。但是作者没说为什么非要取最大的k个数。原文链接：[Large-Scale
  Learnable Graph Convolutional Networks](https://arxiv.org/abs/1808.03965?context=stat.ML)
draft: false
math: true
tags:
- deep learning
- Graph
- graph convolutional network
- large-scale learning
title: Large-Scale Learnable Graph Convolutional Networks
---
KDD 2018.将图结构数据变换到网格状数据中，使用传统的一维卷积进行卷积。变换的方式是：针对每个特征的大小，对邻居结点进行排序，取这个特征前k大的数作为它邻居这列特征的k个值。如果邻居不够，那就用0来补。这样就能得到该顶点的邻居信息，组成一个矩阵，然后使用一维卷积。但是作者没说为什么非要取最大的k个数。原文链接：[Large-Scale Learnable Graph Convolutional Networks](https://arxiv.org/abs/1808.03965?context=stat.ML)
<!--more-->

# 摘要
卷积神经网络在网格数据上取得了很大的成功，但是在学习像图这样的数据的时候就面临着很多的挑战。CNN中，可学习的局部滤波器可以自动地捕获高层次的特征。滤波器的计算需要感受野内有固定数量的单元。然而，在图结构中，邻居单元的数量不固定，而且邻居也不有序，所以阻碍了卷积的操作。我们提出了可学习图卷积层(learnable graph convolutional layer LGCL)来解决这些挑战。基于值的排序，LGCL为每个特征自动地选择固定数量的邻居结点，以此将图结构数据变换到1维的网格结构中，然后就可以在图上使用常规的卷积操作了。为了能让模型在大尺度的图上训练，我们提出了一个子图训练方法来减少过多的内存和计算资源的开销。在顶点分类任务上，不论是transductive 还是 inductive，表现得都更好一些。我们的结果展示出了我们的子图训练方法比前人的方法更高效。

# 3. methods

## 3.1 Challenges of Applying Convolutional Operations on Graph Data

为了让传统的卷积操作可以应用在图上，需要解决两个图结构数据和网格数据的差异。首先，顶点的邻居数量通常会变化。其次，我们不能对邻居顶点进行排序，因为他们没有可供排序的信息。举个例子，社交网络中，每个人都可以看作是一个顶点，边表示人与人之间的关系。显然，每个顶点的邻居顶点数量是不同的，因为人们可以有不同数量的朋友。而且，如果没有额外的信息，很难对他们进行排序。

网格数据可以看作是一种特殊的图结构数据，每个顶点有固定数量的邻居。因为卷积操作是直接应用在图像这样的网格数据上。为了看清楚固定邻居数量以及排序信息的重要性，我们举个例子，有一个$3 \times 3$的卷积核，扫描一张图像。我们将这张图片考虑成一个特殊的图，每个像素是一个顶点。在扫描的过程中，计算包括了中心结点和周围8个邻居结点的计算。这8个顶点在这个特殊的图中通过边连接到中心结点。与此同时，我们使用他们和中心结点的相对位置对他们排序，这对于卷积操作很重要，因为在扫描的过程中，滤波器的权重和图中的顶点要一一对应。举个例子，在上面的例子中，$3 \times 3$的卷积核，左上角的权重应该总是对应中心节点左上方的邻居结点。没有这样的排序信息，卷积的输出结果就不再是确定的。从刚才的讨论中可以看到传统卷积在图结构数据上应用的挑战。为了解决这两个挑战，我们提出了一个方法将图结构数据变换到网格数据内。

## 3.2 learnable Graph Convolutional Layers
为了让传统卷积可以在图上可用，我们提出了LGCL。LGCL的layer-wise传播规则写为：

$$\tag{3}
\tilde{X}\_l = g(X\_l, A, k),\\
X\_{l+1} = c(\tilde{X}\_l)
$$

其中，$A$是邻接矩阵，$g(\cdot)$使用了$k$-largest Node Selection，将图结构数据映射到网格结构，$c(\cdot)$表示1维常规的CNN，将顶点信息聚合，为每个顶点输出了一个新的特征向量。我们会在下面分开讨论$g(\cdot)$和$c(\cdot)$。

**$k$-largest Node Selection**. 我们提出了一个新的方法称为$k$-largest Node Selection，将图结构映射到网格数据上，其中$k$是LGCL的超参数。在这个操作之后，每个顶点的邻居信息聚合，表示成一个有$(k+1)$个位置的1维的网格状。变换后的数据会输入到CNN中来生成新的特征向量。

假设有行向量$x^1\_l, x^2\_l, ..., x^N\_l$的$X\_l \in \mathbb{R}^{N \times C}$，表示$N$个顶点的图，每个顶点有$C$个特征。邻接矩阵$A \in \mathbb{N}^{N \times N}$，$k$为定值。顶点$i$的特征向量是$x^i\_l$，它有$n$个邻居。通过在$A$中的一个简单查找，我们可以获得这些邻居结点的下标，$i\_1, i\_2, ..., i\_n$。对它们对应的特征向量$x^{i\_1}\_l, x^{i\_2}\_l, ..., x^{i\_n}\_l$进行拼接，得到$M^i\_l \in \mathbb{R}^{n \times C}$。假设$n \geq k$，就没有泛化上的损失。如果$n < k$，我们可以使用全为0的列，给$M^i\_l$加padding。$k$-largest node selection是在$M^i\_l$上做的：也就是，对于每列，我们排出$n$个值，然后选最大的$k$个数。我们就可以得到一个$k \times C$的输出矩阵。因为$M^i\_l$表示特征，这个操作等价于为每个特征选择$k$个最大值。通过在第一行插入$x^i\_l$，输出变为$\tilde{M}^i\_l \in \mathbb{R}^{(k+1) \times C}$。如图2左部分。通过对每个顶点重复这个操作，$g(\cdot)$将$X\_l$变为$\tilde{X}\_l \in \mathbb{R}^{N \times (k + 1) \times C}$。

![Figure2](/blog/images/large-scale-learnable-graph-convolutional-networks/Fig2.JPG)

注意，如果将$N$，$(k+1)$，$C$分别看作是batch size，spatial size，通道数，那么$\tilde{X}\_l$可以看作是1维网格状的结构。因此，$k$个最大顶点选择函数$g(\cdot)$成功地将图结构变换为网格结构。这个操作充分利用了实数的自然顺序信息，使得每个顶点有固定数量的有序邻居。

**1-D Convolutional Neural Networks**. 就像3.1节讨论的，传统的卷积操作可以直接应用到网格状的数据上。$\tilde{X}\_l \in \mathbb{R}^{N \times (k + 1) \times C}$是1维的数据，我们部署一个一维CNN模型$c(\cdot)$。LGCL基本的功能是聚合邻居信息，为每个顶点更新特征。后续的话，它需要$X\_{l + 1} \in \mathbb{R}^{N \times D}$，其中$D$是更新后的特征空间的维度。一维CNN $c(\cdot)$ 使用$\tilde{X}\_l \in \mathbb{R}^{N \times (k + 1) \times C}$作为输入，输出一个$N \times D$的矩阵，或是$N \times 1 \times D$的矩阵。$c(\cdot)$可以将空间维度从$(k+1)$减小到$1$。

注意，$N$看作是batch size，与$c(\cdot)$的设计无关。结果就是，我们只聚焦于一个样本，也就是图中的一个顶点。对于顶点$i$，变换得到的输出是$\tilde{M}^i\_l \in \mathbb{R}^{(k + 1) \times C}$，是$c(\cdot)$的输入。由于任何一个卷积核大于1且没有padding的卷积都会减少空间的大小，最简单的$c(\cdot)$只有一个卷积核大小为$(k+1)$的卷积，没有padding。输入和输出的通道数分别为$C$和$D$。同时，可以部署任意一个多层CNN，得到最后的输出的维度是$1 \times D$。图2右侧展示了一个两层CNN的例子。再对所有的$N$个顶点使用一次$c(\cdot)$，输出$X\_{l+1} \in \mathbb{R}^{N \times D}$。总结一下，我们的LGCL使用$k$最大顶点选择以及传统的一维CNN，将图结构变换到网格数据，实现了对每个顶点进行的特征聚合和特征过滤。

## 3.3 可学习的图卷积网络

越深的网络一般会产生越好的结果。然而，之前在图上的深度模型，如GCN，只有两层。尽管随着深度的增加，它们的性能有有所下降[Kipf & Welling 2017]，我们的LGCL可以构造的很深，构造出图顶点分类的可学习的图卷积网络。我们基于densely connected convolutional networks(DCNNs)，构造了LGCNs，前者获得了ImageNet分类任务最好的成绩。

在LGCN中，我们先用一个图嵌入层来生成顶点的低维表示，因为原始输入一般都是高维特征，比如Cora数据集。第一层的图嵌入层本质上就是一个线性变换，表示为：

$$\tag{4}
X\_1 = X\_0 W\_0
$$

其中，$X\_0 \in \mathbb{R}^{N \times C\_0}$表示高维的输入，$W\_0 \in \mathbb{R}^{C\_0 \times C\_1}$将特征空间从$C\_0$映射到了$C\_1$。结果就是，$X\_1 \in \mathbb{R}^{N \times C\_1}$和$C\_1 < C\_0$。或者，使用一个GCN层来做图嵌入。如第二部分描述的，GCN层中的参数数量等价于传统的图嵌入层中参数的数量。

在图嵌入层后，我们堆叠多个LGCL，多少个取决于数据的复杂程度。因为每个LGCL只能聚合一阶邻居的信息，也就是直接相连的邻居顶点，堆叠LGCL可以从一个更大的顶点集中获得信息，这也是传统CNN的功能。为了提升模型的性能，帮助训练过程，我们使用skip connections来拼接LGCL的输入和输出。最后，在softmax激活前使用一个全连接层。

就像LGCN的设计理念，$k$以及堆叠的LGCL的数量是最重要的超参数。顶点的平均度是选择$k$的一个重要参数。LGCL的数量应该依赖任务的复杂度，比如类别的个数，图的顶点数等。越复杂的模型需要越深的模型。

## 3.4 Sub-Graph Training on Large-Scale Data

大部分图上的深度学习模型都有另一个限制。在训练的时候，输入的是所有顶点的特征向量以及整个图的邻接矩阵。图的尺寸大的时候这个矩阵就会变大。这些方法在小尺度的图上表现的还可以。但是对于大尺度的图，这些方法一般都会导致内存和计算资源极大的开销，限制了这些模型的一些应用。

其他类型的数据集也有相似的问题，比如网格数据。举个例子，图像分割上的深度模型通常使用随机切片的方式来处理大的图片。受到这种策略的启发，我们随机的将图“切分”，使用得到的小图进行训练。然而，尽管一张图片的一个矩形部分很自然地包含了像素的邻居信息。如何处理图中顶点的不规则连接还是一个问题。

我们提出了子图选择算法来解决大尺度图上计算资源的问题，如算法1所示。给定一个图，我们先采样出一些初始顶点。从它们开始，我们使用广度优先搜索算法，迭代地将邻接顶点扩充到子图内。经过一定次数的迭代后，初始顶点的高阶邻居顶点就会被加进去。注意，我们在算法1中使用一个简单的参数$N\_m$。实际上在每个迭代中，我们将$N\_m$设置为了不同的值。图4给出了子图选择过程的一个例子。

![Algorithm1](/blog/images/large-scale-learnable-graph-convolutional-networks/Alg1.JPG)

![Figure4](/blog/images/large-scale-learnable-graph-convolutional-networks/Fig4.JPG)

这样随机的切分子图，我们可以在大尺度的图上训练深层模型。此外，我们可以充分利用mini-batch训练方法来加速学习过程。在每轮训练中，我们可以使用子图训练方法采样多个子图，然后把它们放到batch中。对应的特征向量和邻接矩阵组成了网络的输入。

# 4. Experimental studies

代码：[https://github.com/divelab/lgcn/](https://github.com/divelab/lgcn/)

## 4.2 Experimental Setup

**Transduction Learning.** 在transductive learning 任务中，我们像图3一样部署LGCN模型。因为transductive learning数据集使用高维的词袋表示作为顶点的特征向量，输入通过一个图嵌入层来降维。我们这里使用GCN层作为图嵌入层。