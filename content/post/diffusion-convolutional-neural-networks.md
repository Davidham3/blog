---
categories:
- 论文阅读笔记
date: 2018-07-19 11:17:40+0000
description: NIPS 2016。DCNNs，写的云里雾里的，不是很懂在干什么。。。就知道是融入了转移概率矩阵，和顶点的特征矩阵相乘，算出每个顶点到其他所有顶点的
  $j$ 步转移的特征与转移概率的乘积，成为新的顶点表示，称为diffusion-convolutional representation，然后乘以一个卷积核，套一个激活，卷积就定义好了。应用还是在顶点分类与图分类上。原文链接：[Diffusion-Convolutional
  Neural Networks](https://arxiv.org/abs/1511.02136)。
draft: false
math: true
tags:
- deep learning
- Graph
- graph convolutional network
title: Diffusion-Convolutional Neural Networks
---
NIPS 2016。DCNNs，写的云里雾里的，不是很懂在干什么。。。就知道是融入了转移概率矩阵，和顶点的特征矩阵相乘，算出每个顶点到其他所有顶点的 $j$ 步转移的特征与转移概率的乘积，成为新的顶点表示，称为diffusion-convolutional representation，然后乘以一个卷积核，套一个激活，卷积就定义好了。应用还是在顶点分类与图分类上。原文链接：[Diffusion-Convolutional Neural Networks](https://arxiv.org/abs/1511.02136)。
<!--more-->
# 摘要
我们提出了diffusion-convolutional neural networks(DCNNs)，是图结构数据上的新模型。引入diffusion-convolution操作，可以从图结构数据中得到基于扩散性质的表示，用于顶点分类。DCNN还有几个性质，关于一个图数据的隐含表示，还有多项式时间复杂度的预测以及高效的GPU实现。通过多个真实数据集的实验，DCNN表现出了在关系顶点分类任务上超越概率关系模型以及kernel-on-graph的结果。

# 1 引言
处理结构化数据很难。一方面是找到正确的方式表示并挖掘数据的结构可以提升预测的性能；另一方面，找到这样的表示很难，增加结构信息到模型中会急剧地增加预测和学习的复杂度。

我们的工作是为一类结构化数据设计一个灵活的模型，这个模型增强预测能力且避免复杂度的增加。为了完成这个模型，我们通过引入"diffusion-convolution"操作将卷积神经网络扩展到图结构数据上。简单地说，并不是像标准卷积操作一样扫描一个矩形的参数，diffusion-convolution操作通过扩散性的过程扫描图结构输入中每个顶点，构建一个隐含表示。

我们的受到的启发是：捕获了图扩散性的表示在预测上可以提供比图本身更好的结果。图扩散性可以表达成矩阵的幂序列，提供了一个简单的机制来包含关于实体的上下文信息，这些实体可以在多项式时间复杂度内计算出来，并在GPU上实现。

我们提出了diffusion-convolutional神经网络(DCNN)，在图数据的各种各样的分类任务上测试了性能。在分类任务中很多技术包含了结构的信息，比如概率关系模型和核方法；DCNN提供了一个补充的方法，在顶点分类上获得了巨大的提升。

DCNN的优势：
·**精度：** DCNN比其他方法在顶点分类任务上精度更高，图分类上表现的也不错。
·**灵活性：** DCNN提供了图数据的灵活表示，使用简单的处理对顶点特征、边特征以及结构信息进行编码。DCNN可以用于很多分类任务，包括顶点分类，边分类，图分类。
·**速度：** DCNN的预测可以表示成一系列多项式时间复杂度的tensor操作，模型可以在GPU上实现。

# 2 模型
长度为 $T$ 的一个图的集合 $\mathcal{G} = \lbrace G\_t \mid t \in 1...T \rbrace $。每个图 $G\_t = (V\_t, E\_t)$ 由顶点 $V\_t$ 和边 $E\_t$ 组成。顶点一起表示为一个 $N\_t \times F$ 的特征矩阵 $X\_t$，其中 $N\_t$ 是 $G\_t$ 的顶点数，边 $E\_t$ 通过一个 $N\_t \times N\_t$ 的邻接矩阵 $A\_t$ 编码，通过这个我们可以计算出一个度归一化的转移矩阵 $P\_t$，这个矩阵给出了从顶点 $i$ 一步转移到 $j$ 的概率。图 $G\_t$ 没有限制，有向无向，带权不带权都可以。对于我们的任务来说，要么是顶点、边有标签 $Y$，要么是图有标签 $Y$，不同情况下 $Y$ 的维度不同。

如果 $T=1$，也就是只有一个图，标签是顶点或边，那么预测标签 $Y$ 就转换为了半监督分类问题了；如果输入中没有边的表示，就变成了标准的监督问题。如果 $T>1$，标签是每个图的标签，那就是监督图分类问题。

![Figure1](/blog/images/diffusion-convolutional-neural-networks/Fig1.JPG)

DCNN接受 $\mathcal{G}$ 作为输入，返回一个 $Y$ 的hard prediction或是条件概率分布 $\mathbb{P}(Y \mid X)$。每个实体（顶点、图、或边）被转换为扩散卷积表示，由 $F$ 个特征上 $H$ 步扩散的维度为 $H \times F$ 的实数矩阵定义，每个实体是由 $H \times F$ 的实数矩阵 $W^c$ 和一个非线性可微分函数 $f$ 计算激活定义的。所以对于顶点分类任务，图 $t$ 的扩散卷积(diffusion-convolutional)表示 $Z\_t$，是一个 $N\_t \times H \times F$ 的tensor，如图1a所示；对于图或边分类任务，$Z\_t$ 是一个 $H \times F$ 或 $N\_t \times H \times F$ 的矩阵，如图1b和图1c。(原文里面写的是 $M\_t \times H \times F$，我觉得是写错了)

术语"diffusion-convolution"的意思是唤起卷积神经网络的特征的特征：feature learning, parameter tying, invariance。DCNN核心操作是从顶点和他们的特征映射到从那个顶点开始的扩散过程的结果上。不同于标准的CNN，DCNN参数根据搜索的深度而不是他们在网格中的位置而绑定起来。扩散卷积表示对于顶点的index是不变的，而不是他们的位置；换句话说，两个同质的图的扩散卷积激活会是一样的。不像标准的CNN，DCNN没有池化。

**顶点分类** $P^\ast\_t$ 是一个 $N\_t \times H \times N\_t$ 的tensor，包含了 $P\_t$ 的幂序列，对顶点 $i$，$j$ 步，图 $t$ 的特征 $k$ 的扩散卷积的激活值 $Z\_{tijk}$ 是：
$$\tag{1}
Z\_{tijk} = f(W^c\_{jk} \cdot \sum^{N\_t}\_{l=1}P^*\_{tijl}X\_{tlk})
$$

激活使用矩阵形式可以写成：
$$\tag{2}
Z\_t = f(W^c \odot P^\ast\_t X\_t)
$$

其中 $\odot$ 表示element-wise乘法；图1a。模型只有 $O(H \times F)$ 个参数，使得隐扩散卷积表示的参数数量与输入大小无关。

模型通过一个连接 $Z$ 和 $Y$ 的dense layer完成。对于 $Y$ 的hard prediction，表示为 $\hat{Y}$，可以通过最大的激活值得到，条件概率分布 $\mathbb{P}(Y \mid X)$ 可以通过使用softmax得到：
$$\tag{3}
\hat{Y} = \arg\max(f(W^d \odot Z))
$$
$$\tag{4}
\mathbb{P}(Y \mid X) = \mathrm{softmax}(f(W^d \odot Z))
$$

**图分类** DCNN可以通过在顶点上取均值激活扩展成图分类
$$\tag{5}
Z\_t = f(W^c \odot 1^T\_{N\_t} P^\ast\_t X\_t / N\_t)
$$
其中 $1\_{N\_t}$ 是一个 $N\_t \times 1$ 的向量，如图1b所示。

**学习** DCNN使用随机梯度下降训练。每轮，顶点的index随机的分到几个batches中。每个batch的error通过取taking slices of the graph definition power series，然后正向、反向、梯度上升更新权重。也使用了windowed early stopping，如果validation error大于前几轮的平均值，就stop。