---
categories:
- 论文阅读笔记
date: 2018-10-31 21:58:41+0000
description: AAAI 2018。这篇论文很有趣，讲的是 GCN 堆得过多了之后，效果会变差的问题。作者分析了一下为什么会变差，主要是因为 GCN 的本质实际上是对每个结点的邻居特征和自身特征做线性组合，权重和邻接矩阵相关，所以对于顶点分类问题来说，如果堆得层数多了，就会让一个结点的特征聚合越来越多邻居的特征，让大家都变得相似，从而使得类间的相似度增大，自然分类效果就差了。作者提出了两个方法解决这个问题，算训练上的
  trick 吧。原文链接：[Deeper Insights into Graph Convolutional Networks for Semi-Supervised
  Learning](https://arxiv.org/abs/1801.07606)
draft: false
math: true
tags:
- deep learning
- machine learning
- graph convolutional network
- Graph
title: Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning
---
AAAI 2018。这篇论文很有趣，讲的是 GCN 堆得过多了之后，效果会变差的问题。作者分析了一下为什么会变差，主要是因为 GCN 的本质实际上是对每个结点的邻居特征和自身特征做线性组合，权重和邻接矩阵相关，所以对于顶点分类问题来说，如果堆得层数多了，就会让一个结点的特征聚合越来越多邻居的特征，让大家都变得相似，从而使得类间的相似度增大，自然分类效果就差了。作者提出了两个方法解决这个问题，算训练上的 trick 吧。原文链接：[Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning](https://arxiv.org/abs/1801.07606)
<!--more-->

# Abstract

机器学习中很多有趣的问题正在用深度学习工具来重新审视。对于基于图的半监督学习问题，最新的一个重要进展就是图卷积神经网络 (GCNs)，这个模型可以很好的将顶点局部特征和图的拓扑结构整合进卷积层内。尽管 GCN 模型和其他的 state-of-the-art 方法相比效果更好，但是它的机理目前还不是很清楚，而且需要很多的标记数据用于验证以及模型选择。

在这篇论文中，我们深入了 GCN 模型，解决了它的底层限制。首先，我们发现 GCN 模型的图卷积实际上是一个拉普拉斯平滑的特殊形式，这是 GCN 工作的关键原因，但是这也会给很多卷积层带来潜在的危害。其次，为了克服 GCN 层数少的限制，我们提出了协同训练和自训练方法来训练 GCNs。我们的方法显著地提升了 GCNs 在标记样本少的情况下的学习，并且让他们避免了使用额外的标记用来验证。大量的实验证明了我们的理论和方案。

# 1 Introduction

深度学习中的突破使得人工智能和机器学习中正在发生范式变化。一方面，很多老问题通过深度神经网络重新审视，很多原来看起来在任务中无法完成的巨大进步现在也在发生着，如机器翻译和计算机视觉。另一方面，像几何深度学习 (Bronstein et al. 2017) 这样的技术正在发展，可能会将深度神经模型泛化到新的或非传统的领域。

众所周知，深度学习模型一般需要大量的标记数据，在很多标记训练数据代价很大的场景就无法满足这样的要求。为了减少用于训练的数据的数量，最近的研究开始关注 few-shot learning (Lake, Salakhutdinov, and Tenenbaum 2015; Rezende et al. 2016)——从每个类只有很少的样本中学习一个分类模型。和 few-shot learning 相近的是半监督学习，其中有大量的未标记样本可以用来和很少量的标记样本一起用于训练。

很多研究者已经证实了如果使用恰当，在训练中利用未标记样本可以显著地提升学习的精度 (Zhu and Goldberg 2009)。关键问题是最大化未标记样本的结构和特征信息的有效利用。由于强力的特征抽取能力和深度学习近些年的成功案例，已经有很多人使用基于神经网络的方法处理半监督学习，包括 ladder network (Rasmus et al. 2015), 半监督嵌入 (Weston et al. 2008)，planetoid (Yang, Cohen, and Salakhutdinov 2016)，图卷积网络 (Kipf and Welling 2017)。

最近发展的图卷积神经网络 (GCNNs) (Defferrard, Bresson, and Vandergheynst 2016) 是一个将欧氏空间中使用的卷积神经网络 (CNNs) 泛化到对图结构数据建模的成功尝试。在他们的初期工作 (Kipf and Welling 2017)，Kifp and Welling 提出了一个 GCNNs 的简化类型，称为图卷积网络 (GCNs)，应用于半监督分类。GCN 模型很自然地将图结构数据的连接模式和特征属性集成起来，而且比很多 state-of-the-art 方法在 benchmarks 上好很多。尽管如此，它也有很多其他基于神经网络的模型遇到的问题。用于半监督学习的 GCN 模型的工作机理还不清楚，而且训练 GCNs 仍然需要大量的标记样本用于调参和模型选择，这就和半监督学习的理念相违背。

在这篇论文中，我们弄清楚了用于半监督学习的 GCN 模型。特别地，我们发现 GCN 模型中的图卷积是拉普拉斯平滑的一种特殊形式，这个平滑可以混合一个顶点和它周围顶点的特征。这个平滑操作使得同一类簇内顶点的特征相似，因此使分类任务变得简单，这使为什么 GCNs 表现的这么好的关键原因。然而，这也会带来 over-smoothing 的问题。如果 GCN 有很多卷积层后变深了，那么输出的特征可能会变得过度平滑，且来自不同类簇的顶点可能变得无法区分。这种混合在小的数据集，且只有很少的卷积层上发生的很快，就像图2展示的那样。而且，给 GCN 模型增加更多的层也会使它变得难以训练。

然而，一个浅层的 GCN 模型，像 Kipf & Welling 2017 使用的两层 GCN 有它自身的限制。除此以外它还需要很多额外的标记用来验证，它也会遇到卷积核局部性等问题。当只有少数标记的时候，一个浅层的 GCN 模型不能有效的将标记传播到整个图上。如图1所示，GCNs 的表现会随着训练集的减少急速下降，甚至有500个额外标记用来验证。

为了克服限制并理解 GCN 模型的全部潜能，我们提出了一种协同训练方法和一个自训练方法来训练 GCNs。通过使用随机游走模型来协同训练一个 GCN，随机游走模型可以补充 GCN 模型在获取整个图拓扑结构上的能力。通过自训练一个 GCN，我们可以挖掘它的特征提取能力来克服它的局部特性。融合协同训练和自训练方法可以从本质上提升 GCN 模型在半监督学习上只有少量标记的效果，而且使它不用使用额外的标记样本用来验证。如图1所示，我们的方法比 GCNs 好了一大截。

总而言之，这篇论文的关键创新有：1) 对半监督学习的 GCN 模型提供了新的视角和新的分析；2) 提出了对半监督学习的 GCN 模型提升的解决方案。

# 2 Preliminaries and Related Works

首先，我们定义一些符号。图表示为 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，其中 $\mathcal{V}$ 是顶点集，$\vert \mathcal{V} \vert = n$，$\mathcal{E}$ 是边集。在这篇论文中，我们考虑的是无向图。$A = [a\_{ij}] \in \mathbb{R}^{n \times n}$ 是邻接矩阵，且为非负的。$D = \mathrm{diag}(d\_1, d\_2, ..., d\_n)$ 表示度矩阵，$d\_i = \sum\_j a\_{ij}$ 是顶点 $i$ 的度。图拉普拉斯矩阵 (Chung 1997) 定义为 $L := D - A$，归一化的图拉普拉斯矩阵的两个版本分别定义为：$L\_{sym} := D^{-\frac{1}{2}} L D^{-\frac{1}{2}}$ 和 $L\_{rw} := D^{-1}L$。

**Graph-Based Semi-Supervised Learning**

这篇论文中我们考虑的问题是图上的半监督分类任务。给定一个图 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, X)$，其中 $X = \mathrm{[x\_1, x\_2, ..., x\_n]^T} \in R^{n \times c}$ 是特征矩阵，$\mathrm{x}\_i \in R^c$ 是顶点 $i$ 的 $c$ 维特征向量。假设给定了一组顶点 $\mathcal{V}\_l$ 的标记，目标是预测其余顶点 $\mathcal{V}\_u$ 的标记。

基于图的半监督学习在过去的二十年成为了一个流行的研究领域。通过挖掘图或数据的流形结构，是可以通过少量标记进行学习的。很多基于图的半监督学习方法形成了类簇假设 (cluster assumption) (Chapelle and Zien 2005)，假设了一个图上临近的顶点倾向于有共同的标记。顺着这条路线的研究包括 min-cuts (Blum and Chawla 2001) 和 randomized min-cuts (Blum et al. 2004)，spectral graph transducer (Joachims 2003)，label propagation (Zhu, Ghahramani, and Lafferty 2003) and its variants (Zhou et al. 2004; Bengio, Delalleau, and Le Roux 2006)，modified adsorption (Talukdar and Crammer 2009)，还有 iterative classification algorithm (Sen et al. 2008)。

但是图只表示数据的结构信息。在很多应用，数据的样本是以包含信息的特征向量表示，而不是在图中表现。比如，在引文网络中，文档之间的引用链接描述了引用关系，但是文档是由 bag-of-words 向量表示的，这些向量描述的内容是文档的内容。很多半监督学习方法寻求对图结构和数据的特征属性共同建模。一个常见的想法是使用一些正则项对一个监督的学习器进行正则化。比如，manifold regularization (LapSVM) (Belkin, Niyogi, and Sindhwani 2006) 使用一个拉普拉斯正则项对 SVM 进行正则化。深度半监督嵌入 (Weston et al. 2008) 使用一个基于嵌入的正则项对深度神经网络进行正则化。Planetoid (Yang, Cohen, and Salakhutdinov 2016) 也通过共同地对类标记和样本的上下文预测对神经网络进行正则化。

**Graph Convolutional Networks**

图卷积神经网络 (GCNNs) 将传统的卷积神经网络泛化到图域中。主要有两类 GCNNs (Bronstein et al. 2017): spatial GCNNs 和 spectral GCNNs。空间 GCNNs 将卷积看作是 "patch operator"，对每个顶点使用它的邻居信息构建新的特征向量。谱 GCNNs 通过对图信号 $\bf{s} \in \mathcal{R}^n$ 在谱域上进行分解，然后使用一个在谱成分上的谱卷积核 $g\_\theta$ (是 $L\_{sym}$ 的特征值的一个函数) (Bruna et al. 2014; Sandryhaila and Moura 2013; Shuman et al. 2013)。然而这个模型需要计算出拉普拉斯矩阵的特征向量，这对于大尺度的图来说是不太实际的。一种缓解这个问题的方法是通过将谱卷积核 $g\_\theta$ 通过切比雪夫多项式趋近到 $K^{th}$ 阶 (Hammond, Vandergheynst, and Gribonval 2011)。在 (Defferrard, Bresson, and Vandergheynst 2016)，Defferrard et al. 使用这个构建了 $K$ 阶 ChebNet，卷积定义为：

$$\tag{1}
g\_\theta \star \mathbf{s} \approx \sum^K\_{k=0} \theta'\_k T\_k (L\_{sym}) \mathbf{s},
$$

其中 $\bf{s} \in \mathcal{R}^n$ 是图上的信号，$g\_\theta$是谱滤波器，$\star$ 是卷积操作，$T\_k$ 是切比雪夫多项式，$\theta' \in \mathcal{R}^K$ 是切比雪夫系数向量。通过这种趋近，ChebNet 域谱无关。

在 (Kipf and Welling 2017) 中，Kipf and Welling 将上面的模型通过让 $K = 1$ 进行了简化，将 $L\_{sym}$ 的最大特征值趋近为2.在这种形式中，卷积变成：

$$\tag{2}
g\_\theta \star \mathbf{s} = \theta(I + D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) \mathbf{s},
$$

其中 $\theta$ 是切比雪夫系数。然后对卷积矩阵使用一种正则化的技巧：

$$\tag{3}
I + D^{-\frac{1}{2}} A D^{-\frac{1}{2}} \rightarrow \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}},
$$

其中 $\tilde{A} = A + I$，$\tilde{D} = \sum\_j \tilde{A}\_{ij}$.

将卷积泛化到带有 $c$ 个通道的图信号上，也就是 $X \in \mathcal{R}^{n \times c}$ (每个顶点是一个 $c$ 维特征向量)，使用 $f$ 谱卷积核，简化后的模型的传播规则是：

$$\tag{4}
H^{(l + 1)} = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} \Theta^{(l)}),
$$

其中，$H^{(l)}$ 是第 $l$ 层的激活值矩阵，$H^{(0)} = X$，$\Theta^{(l)} \in \mathcal{R}^{c \times f}$ 是第 $l$ 层可训练的权重矩阵，$\sigma$ 是激活函数，比如 $ReLU(\cdot) = max(0, \cdot)$。

这个简化的模型称为图卷积网络 (GCNs)，是我们这篇论文关注的重点。

**Semi-Supervised Classification with GCNs**

在 Kipf and Welling 2017 中，GCN 模型以一种优雅的方式做半监督分类任务。模型是一个两层 GCN，在输出时使用一个 softmax：

$$\tag{5}
Z = \mathrm{softmax}(\hat{A} ReLU (\hat{A} X \Theta^{(0)}) \Theta^{(1)} ),
$$

其中 $\hat{A} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$，$\mathrm{softmax}(x\_i) = \frac{1}{\mathcal{Z}} exp(x\_i)$，$\mathcal{Z} = \sum\_i exp(x\_i)$。损失函数是所有标记样本上的交叉熵：

$$\tag{6}
\mathcal{L} := - \sum\_{i \in \mathcal{V}\_l} \sum^F\_{f=1} Y\_{if} \mathrm{ln} Z\_{if},
$$

其中 $\mathcal{V}\_l$ 是标记顶点的下标，$F$ 是输出特征的维数，等价于类别数。$Y \in \mathcal{R}^{\vert \mathcal{V}\_l \vert \times F}$ 是标记矩阵。权重参数 $\Theta^{(0)}$ 和 $\Theta^{(1)}$ 可以通过梯度下降训练。

GCN 模型在卷积中自然地融合了图的结构和顶点的特征，未标记的顶点的特征和临近的标记顶点的混合在一起，然后通过多个层在网络上传播。GCNs 在 Kipf & Welling 2017 中比很多 state-of-the-art 方法都好很多，比如在引文网络上。

# 3 Analysis

尽管它的性能很好，但是用于半监督学习的 GCN 模型的机理还没有弄明白。在这部分我们会走近 GCN 模型，分析它为什么好使，并指出它的限制。

**Why GCNs Work**

我们将 GCN 和最简单的全连接神经网络 (FCN) 进行比较，传播规则是：

$$\tag{7}
H^{(l + 1)} = \sigma(H^{(l)} \Theta^{(l)}).
$$

GCN 和 FCN 之间的唯一一个区别是图卷积矩阵 $\hat{A} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$ (式5)用在特征矩阵 $X$ 的左边。我们在 Cora 数据集上，每类 20 个 标签，做了半监督分类的测试。如表1所示。即便是只有一层的 GCN 也比一层的 FCN 好很多。

![Table1](/images/deeper-insights-into-graph-convolutional-networks-for-semi-supervised-learning/Table1.JPG)

**Laplacian Smoothing.** 考虑一个一层的 GCN。实际有两步：
1. 从矩阵 $X$ 通过一个图卷积得到新的特征矩阵 $Y$：

$$\tag{8}
Y = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}X.
$$

2. 将新的特征矩阵 $Y$ 放到一个全连接层。很明显，图卷积是性能提升的关键。

我们来自己的检查一下图卷积。假设我们给图中的每个结点增加一个自连接，新的图的邻接矩阵就是 $\tilde{A} = A + I$。输入特征的每个通道的拉普拉斯平滑 (Taubin 1995) 定义为：

$$\tag{9}
\hat{\mathrm{y}}\_i = (1 - \gamma) \mathrm{x}\_i + \gamma \sum\_j \frac{\tilde{a}\_{ij}}{d\_i} \mathrm{x}\_j \quad (\text{for} \quad 1 \leq i \leq n),
$$

其中 $0 < \gamma < 1$ 是控制当前结点的特征和它的邻居的特征之间的权重。我们可以将拉普拉斯平滑写成矩阵形式：

$$\tag{10}
\hat{Y} = X - \gamma \tilde{D}^{-1} \tilde{L} X = (I - \gamma \tilde{D}^{-1} \tilde{L})X,
$$

其中 $\tilde{L} = \tilde{D} - \tilde{A}$。通过设定 $\gamma = 1$，也就是只使用邻居的特征，可得 $\hat{Y} = \tilde{D}^{-1} \tilde{A} X$，也就是拉普拉斯平滑的标准形式。

现在如果我们把归一化的拉普拉斯矩阵 $\tilde{D}^{-1} \tilde{L}$ 替换成对阵的归一化拉普拉斯矩阵 $\tilde{D}^{-\frac{1}{2}} \tilde{L} \tilde{D}^{-\frac{1}{2}}$，让 $\gamma = 1$，可得 $\hat{Y} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X$，这恰好就是式8中的图卷积。我们因此称图卷积是一种特殊形式的拉普拉斯平滑——对称拉普拉斯平滑。注意，平滑仍然会包含顶点特征，因为每个顶点有一个自连接，还有它自己的邻居。

拉普拉斯平滑计算了顶点的新的特征，也就是顶点自身和邻居的加权平均。因为同一类簇的顶点倾向于连接的更紧密，这使得分类任务变得更简单。因为我们可以从表1看出只使用一次平滑就很有效了。

**Multi-layer Structure.** 我们可以从表1看出尽管两层的 FCN 比 一层的 FCN 有了些许的提升，两层的 GCN 却比 一层的 GCN 好了很多。这是因为在第一层的激活值上再使用平滑使得同一个类簇中的顶点特征变得更像四了，使分类任务更简单。

**When GCNs Fail**

我们已经证明了图卷积本质上就是一种拉普拉斯平滑。那 GCN 中应该放多少层呢？当然不是越多越好。GCN 层多了会不好训练。而且重复使用拉普拉斯平滑可能会混合不同类簇中的顶点的特征，使得他们区分不清。我们来举个例子。

我们在 Zachary 的 karate club dataset (Zachary 1977) 上跑几个层数不同的模型。这个数据集有 34 个结点，两类，78 条边。GCN 的参数像 (Glorot and Bengio 2010) 中的一样随机初始化。隐藏层的维数是 16，输出层的维度是2。每个结点的特征向量是一个 one-hot 向量。每个 GCN 的输出绘制在图2中。我们可以看到图卷积的影响（图2a）。使用两次平滑，分类效果相对较好。再次使用平滑，点就会混合（图2c，2d，2e）。因为这是个小的数据集，两类之间的顶点有很多连接，所以很快就发生了混合。

接下来，我们会证明重复使用拉普拉斯平滑，顶点的特征以及图的每个连通分量会收敛到相同的值。对于对称的拉普拉斯平滑，他们收敛到的值与顶点度数的二分之一次幂成正比。

假设图 $\mathcal{G}$ 有 $k$ 个连通分量 $\lbrace  C\_i\rbrace ^k\_{i=1}$，对于第 $i$ 个连通分量的指示向量表示为 $\mathbf{1}^{(i)} \in \mathbb{R}^n$。这个向量表示顶点是否在分量 $C\_i$中，即：

$$\tag{11}
\mathbf{1}^{(i)}\_j = \begin{cases}
1, v\_j \in C\_i,\\
0, v\_j \notin C\_i
\end{cases}
$$

**Theorem 1. ** 如果一个图没有二分的连通分量，那么对于任意 $\mathrm{w} \in \mathbb{R}^n$，$\alpha \in (0, 1]$，

$$
\lim\_{m \rightarrow + \infty} (I - \alpha L\_{rw})^m \mathrm{w} = [\mathbf{1}^{(1)}, \mathbf{1}^{(2)}, ..., \mathbf{1}^{(k)}]\theta\_1,
$$

$$
\lim\_{m \rightarrow + \infty} (I - \alpha L\_{sym})^m \mathrm{w} = D^{-\frac{1}{2}}[\mathbf{1}^{(1)}, \mathbf{1}^{(2)}, ..., \mathbf{1}^{(k)}]\theta\_2,
$$

其中 $\theta\_1 \in \mathbb{R}^k, \theta\_2 \in \mathbb{R}^k$，也就是他们分别收敛到 $\lbrace \mathbf{1}^{(i)}\rbrace ^k\_{i=1}$ 和 $\lbrace  D^{-\frac{1}{2}} \mathbf{1}^{(i)} \rbrace ^k\_{i=1}$。

*Proof.* $L\_{rw}$ 和 $L\_{sym}$ 有相同的 $n$ 个特征值，不同的特征向量 (Von Luxbury 2007)。如果一个图没有二分的连通分量，那么特征值就会在 $[0, 2)$ 区间内 (Chung 1997)。后面就没看懂了。。。

# 4. Solutions

**Co-Train a GCN with a Random Walk Model**

使用一个 partially absorbing random walks (Wu et al. 2012) 来捕获网络的全局结构。方法就是计算归一化的吸收概率矩阵 $P = (L + \alpha \Lambda)^{-1}$，$P\_{i, j}$ 是从顶点 $i$ 出发被吸收到顶点 $j$ 的概率，表示 $i$ 和 $j$ 有多大的可能性属于同一类。然后我们对每类 $k$，计算可信向量 $\mathbf{p} = \sum\_{j \in S\_k} P\_{:, j}$，其中 $\mathbf{p} \in \mathbb{R}^n$，$p\_i$ 是顶点 $i$ 属于类 $k$ 的概率。最后，找到 $t$ 个最可信的顶点把他们加到训练集的类 $k$ 中。

![Alg1](/images/deeper-insights-into-graph-convolutional-networks-for-semi-supervised-learning/Alg1.JPG)

**GCN Self-Training**
另一种方法就是先训练一个 GCN，然后使用这个 GCN 去预测，根据预测结果的 $\text{softmax}$ 分数选择可信的样本，加入到训练集中。

![Alg2](/images/deeper-insights-into-graph-convolutional-networks-for-semi-supervised-learning/Alg2.JPG)