---
categories:
- 论文阅读笔记
date: 2018-07-19 18:53:17+0000
description: NIPS 2017。提出的方法叫 GraphSAGE，针对的问题是之前的 NRL 是 transductive，不能泛化到新结点上，而作者提出的
  GraphSAGE 是 inductive。主要考虑了如何聚合顶点的邻居信息，对顶点或图进行分类。原文链接：[Inductive Representation
  Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
draft: false
math: true
tags:
- deep learning
- Graph
- graph convolutional network
title: Inductive Representation Learning on Large Graphs
---
NIPS 2017。提出的方法叫 GraphSAGE，针对的问题是之前的 NRL 是 transductive，不能泛化到新结点上，而作者提出的 GraphSAGE 是 inductive。主要考虑了如何聚合顶点的邻居信息，对顶点或图进行分类。原文链接：[Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
<!--more-->

# Abstract

现存的方法需要图中所有的顶点在训练 embedding 的时候都出现；这些前人的方法本质是 *transductive*，不能自然地泛化到未见过的顶点上。我们提出了 GraphSAGE，一个 *inductive* 框架，利用顶点特征信息（比如文本属性）来高效地为没有见过的顶点生成 embedding。与其为每个顶点训练单独的 embedding，我们的方法是学习一个函数，这个函数通过从一个顶点的局部邻居采样并聚合顶点特征。我们的算法在三个 inductive 顶点分类数据集上超越了那些很强的 baseline：在 citation 和 Reddit post 数据的演化的信息图中对未见过的顶点分类，在 PPI 多图数据集上可以泛化到完全未见过的图上。

# 1 Introduction

顶点嵌入的基本思想是使用降维技术从高维信息中提炼一个顶点的邻居信息，存到低维向量中。这些顶点嵌入之后会作为后续的机器学习系统的输入，解决像顶点分类、聚类、链接预测这样的问题。

然而，前人的工作专注于对一个固定的图中的顶点进行表示，很多真实的应用需要很快的对未见过的顶点或是全新的图（子图）生成 embedding。这个推断的能力对于高吞吐的机器学习系统来说很重要，这些系统都运作在不断演化的图上，而且时刻都会遇到未见过的顶点（比如 Reddit 上的文章，Youtube 上的用户或视频）。一个生成顶点 embedding 的推断方法也会帮助在拥有同样形式特征的图上进行泛化：举个例子，我们可以从一个有机物得到的 PPI 图上训练一个 embedding 生成器，然后很简单的使用这个模型利用新的有机物上收集的数据生成他们的顶点嵌入。

对比 transductive 问题，推断顶点嵌入问题很困难，因为泛化未见过的顶点需要将新观测到的子图“对齐”到算法已经优化好的顶点嵌入上。一个推断模型必须学习到可以识别一个顶点邻居的结构性质，这个性质既反映了顶点在图中的局部角色，也反映了它的全局位置。

很多现存的生成顶点嵌入的方法是继承于 transductive。这些方法的主流是直接使用矩阵分解目标函数对每个顶点的 embedding 进行优化，因为他们在一个固定的单个图上的顶点做预测，所以不能自然地泛化到未见过的数据。这些方法可以被修改然后在 inductive 问题上运行，但是这些修改往往计算复杂度高，在新的预测之前需要额外的梯度下降优化。当然也有一些在图结构上使用卷积神经网络的方法，在 embedding 上表现的很好。迄今为止，GCN 只在固定的图上的 transductive 问题上应用过。我们工作是扩展了 GCN 到无监督推断任务上，同时还提出了一个方法，可以让 GCN 使用可训练的聚合函数（并不是只有简单的卷积）。

**Present work.** 我们提出了一个通用的框架，叫 GraphSAGE(SAmple and aggreGatE)，用于学习 inductive node embedding。不像基于矩阵分解的嵌入方法，我们利用了顶点特征（比如文本属性，顶点信息，顶点的度）来学习可以生成未见过的顶点的嵌入的函数。通过在算法中结合顶点信息，我们同时学习了每个顶点邻居的拓扑结构和顶点特征在邻居中的分布。尽管我们的研究更专注于富特征的图（如有文本信息的引文网络，有功能/分子组成的生物数据），我们的方法仍能充分利用所有图展现的结构特征（比如顶点的度）。因此我们的算法可以应用在没有顶点特征的图上。

![Figure1](/images/inductive-representation-learning-on-large-graphs/Fig1.JPG)

我们没有对每个顶点都训练一个单独的 embeddding，我们训练了一组 *aggregator functions*，这些函数学习如何从一个顶点的局部邻居聚合特征信息（图1）。每个聚合函数从一个顶点的不同搜索深度聚合信息。测试的时候，或是说推断的时候，我们使用我们训练的系统来对完全未见过的顶点，通过使用学习到的聚合函数来生成 embedding。跟随着前人在生成顶点上的工作，我们设计了无监督的损失函数，使得 GraphSAGE 可以在没有任务监督的情况下训练。我们也展示了如何使用监督的方法训练 GraphSAGE。

我们在三个顶点分类数据集上评估了我们的算法，测试了 GraphSAGE 在未见过的数据上生成有效 embedding 的能力。使用了两个基于 citation 数据和 Reddit post 数据的演化网络（分别预测 paper 和 post 类别），还有一个基于 PPI 的多图泛化实验（预测蛋白质功能）。我们的方法效果很好，跨领域，监督的方法在 F1 值上对比只使用顶点特征的方法平均提高了 51%，而且 GraphSAGE 一直都比 transductive baseline 强很多，尽管 baseline 在未见过的顶点上的运行时间要长 100 倍以上。我们提出的新的聚合结构比受图卷积启发的聚合函数更好（平均提升了 7.4%）。最后我们通过实验证明了我们方法的表达能力，尽管 GraphSAGE 是基于特征的，它却能学习到一个顶点在一个图中的结构信息，（第 5 部分）。

# 2 Related work

**Factorization-based embedding approaches.** 最近的 node embedding 方法使用随机游走的统计和矩阵分解的目标函数。这些方法和传统的方法如谱聚类，multi-dimensional scaling，PageRank 关系很近。因为这些嵌入方法对每个顶点直接训练 embedding，本质上是 transductive，而且需要大量的额外训练（如随机梯度下降）使他们能预测新的顶点。此外，这些方法中的大部分方法，目标函数对于 embedding 的正交变换是不变的，意味着嵌入空间不能自然地在图之间泛化，而且在再次训练的时候会 drift。一个值得注意的例外是 Yang et al. 的 Planetoid-I 算法，是一个 inductive 的方法，基于嵌入的半监督学习。然而，Planetoid-I 在推断的时候不使用任何图结构信息，而在训练的时候将图结构作为一种正则化的形式。不同于前面提到的这些方法，我们是利用特征信息训练可以对未见过的顶点生成 embedding 的模型。

**Supervised learning over graphs.** 除了顶点嵌入方法，还有很多在图结构数据上的监督学习方法。包括很多核方法，图的特征向量从多个图的核得到。最近有很多用于图结构的监督学习的神经网络方法。我们的方法从概念上是受到了这些方法的启发。然而，这些方法试图对整个图（或子图）进行分类，我们的工作关注的是如何对单个顶点生成有效的表示。

**Graph convolutional networks.** 近些年，一些用于图的卷积神经网络被相继提出。这些方法中的大部分不能扩展到大的图上，或者是为了整个图的分类而设计（或是两点都有）。我们的方法与 Kipf et al. 提出的图卷积很相关，GCN 在训练的时候需要整个图的拉普拉斯矩阵。我们算法的一个简单的变体可以看作是 GCN 框架在 inductive setting 上的扩展，我们会在 3.3 说明。

# 3 GraphSAGE

我们的核心思想在于如何从一个顶点的局部邻居聚合特征信息（比如度或近邻顶点的文本特征）。3.1 描述 embedding 的生成算法。3.2 描述随机梯度下降学习参数。

## 3.1 Embedding generation (i.e., forward propagation) algorithm

假设已经学习到了 $K$ 个聚合函数（表示为 $AGGERGATE\_k, \forall k \in \lbrace 1,...,K\rbrace$ ）的参数，对顶点的信息聚合，还有一组权重矩阵 $\mathbf{W}^k, \forall k \in \lbrace 1,...,K\rbrace$，用来在模型的不同层或搜索深度间传播信息。下一节描述参数是怎么训练的。

![Algo1](/images/inductive-representation-learning-on-large-graphs/Alg1.JPG)

算法 1 的思路是在每次迭代，或每一个搜索深度，顶点从他们的局部邻居聚合信息，而且随着这个过程的迭代，顶点会从越来越远的地方获得信息。

算法 1 描述了在各整个图上生成 embedding 的过程，$\mathcal{G} = \left( \mathcal{V}, \Large{\varepsilon} \right)$，以及所有顶点的特征$X\_v, \forall v \in \mathcal{V}$作为输入。在算法 1 最外层循环的每一步如下，$k$ 表示外循环（或搜索深度）的当前步，$\mathbf{h}^k$ 表示当前这步的一个顶点的表示：首先，每个顶点 $v \in \mathcal{V}$聚合了在它在中间邻居的表示，$\lbrace  \mathbf{h}^{k-1}\_u, \forall u \in \mathcal{N}(u) \rbrace$，聚合到向量$\mathbf{h}^{k-1}\_{\mathcal{N}(v)}$中。注意，这个聚合步骤依赖于外循环前一次迭代生成的表示（比如$k - 1$），$k = 0$表示输入的顶点特征。聚合邻居特征向量后，GraphSAGE之后拼接了顶点当前的表示，$\mathbf{h}^{k-1}\_v$，核聚合的邻居向量一起，$\mathbf{h}^{k-1}\_{\mathcal{N}(v)}$，拼接后的向量输入到了激活函数为$\sigma$的全连接层中，将表示变换为下一步使用的形式（$\mathbf{h}^k\_v, \forall v \in \mathcal{V}$）。为了记号的简单，我们将深度为$K$的输出表示记为$\mathbf{z} \equiv \mathbf{h}^K\_v, \forall v \in \mathcal{V}$。邻居表示的聚合可以通过多个聚合架构得到（在算法1中表示为$\mathrm{AGGERGATE}$），我们会在3.3讨论不同的架构。

为了将算法1扩展到minibatch设定上，给定一组输入顶点，我们先采样采出需要的邻居集合（到深度$K$），然后运行内部循环（算法1的第三行），但不是迭代所有的顶点，我们在每个深度只计算必须满足的表示（后记A包括了完整的minibatch伪代码）。

## 3.2 Learning the parameters of GraphSAGE
为了在半监督设定下学习一个有效的表示，我们使用基于图的损失函数来输出表示$\mathbf{z}\_u, \forall u \in \mathcal{V}$，调整权重矩阵$\mathbf{W}^k, \forall k \in \lbrace 1,...,K\rbrace$，聚合函数的参数通过随机梯度下降训练。基于图的损失函数倾向于使得相邻的顶点有相似的表示，尽管这会使相互远离的顶点的表示很不一样：
$$\tag{1}
J\mathcal{G}(\mathbf{z}\_u) = -\log(\sigma(\mathbf{z}^T\_u \mathbf{z}\_v)) - Q \cdot \mathbb{E}\_{v\_n \sim P\_n(v)} \log(\sigma(-\mathbf{z}^T\_u \mathbf{z}\_{v\_n})),
$$
其中$v$是通过定长随机游走得到的$u$旁边的共现顶点，$\sigma$是sigmoid函数，$P\_n$是负采样分布，$Q$定义了负样本的数目。重要的是，不像之前的那些方法，我输入到损失函数的表示$\mathbf{z}\_u$是从包含一个顶点局部邻居的特征生成出来的，而不是对每个顶点训练一个独一无二的embedding（通过一个embedding查询表）。

这个无监督设定模拟了顶点特征提供给后续机器学习应用的情况。在那些表示只在后续的任务中使用的情况下，无监督损失（式1）可以被替换或改良，通过一个以任务为导向的目标函数（比如cross-entropy）。

## 3.3 聚合架构
不像在$N$维网格（如句子、图像、$3\rm{D}$）上的机器学习，一个顶点的邻居是无序的；因此，算法1中的聚合函数必须在以一个无序的向量上运行。理想上来说，一个聚合函数需要是对称的（也就是对它输入的全排列来说是不变的），而且还要可训练，且保持表示的能力。聚合函数的对称性之确保了我们的神经网络模型可以被训练且可以应用于任意顺序的顶点邻居特征集合上。我们检验了三种聚合函数：
**Mean aggregator.** 第一个聚合函数是均值聚合，我们简单的取$\lbrace  \mathbf{h}^{k-1}\_u, \forall v \in \mathcal{N}(v) \rbrace$中的向量的element-wise均值。均值聚合近似等价在transducttive GCN框架[17]中的卷积传播规则。特别地，我们可以通过替换算法1中的4行和5行为以下内容得到GCN的inductive变形：
$$\tag{2}
\mathbf{h}^k\_v \leftarrow \sigma(\mathbf{W} \cdot \mathrm{MEAN}(\lbrace  \mathbf{h}^{k-1}\_v \rbrace \cup \lbrace  \mathbf{h}^{k-1}\_u, \forall u \in \mathcal{N}(v) \rbrace)).
$$
我们称这个修改后的基于均值的聚合器是*convolutional*，因为它是一个粗略的，局部化谱卷积的的线性近似[17]。这个卷积聚合器和我们的其他聚合器的重要不同在于它没有算法1中第5行的拼接操作——卷积聚合器没有将顶点前一层的表示$\mathbf{h}^{k-1}\_v$和聚合的邻居向量$\mathbf{h}^k\_{\mathcal{N}(v)}$拼接起来。拼接操作可以看作一个是在不同的搜索深度或层之间的简单的skip connection的形式，它使得模型获得了巨大的提升。

**LSTM aggregator.** 我们也检验了一个基于LSTM的复杂的聚合器。对比均值聚合器，LSTM有更强的表达能力。然而，LSTM不是对称的这个需要注意，因为他们处理他们的输入是以一个序列的方式。我们简单地将LSTM应用在一个顶点邻居的随机序列上。

**Pooling aggregator.** 我们检验的最后一个聚合器既是对称的，又是可训练的。在这个*池化*方法种，每个邻居的向量都是相互独立输入到全连接神经网络中的；随着这种变化，一个element-wise最大池化操作应用在邻居集合上来聚合信息：
$$\tag{3}
\mathrm{AGGREGATE}^{pool}\_k = \mathrm{max}(\lbrace  \sigma (\mathbf{W}\_{pool} \mathbf{h}^k\_{u\_i} + \mathbf{b}), \forall u\_i \in \mathcal{N}(v) \rbrace),
$$
其中，$\mathrm{max}$表示element-wise最大值操作，$\sigma$是非线性激活函数。原则上，在最大池化使用之前，函数可以是任意的深度多层感知机，但是我们关注的是单个的单层结构。方法是搜到了最近的神经网络架构在学习general point sets上的启发[29]。直觉上来说，多层感知机可以看作是一组函数，这组函数为邻居集合中的每个顶点计算表示。通过对每个计算得到的特征使用最大池化操作，模型有效地捕获了邻居集合的不同方面。注意，原则上，任何对称的向量函数都可以替换$\mathrm{max}$操作器（比如element-wise mean）。我们发现最大池化和均值池化在测试时没有太大的差别，所以使用了最大池化完成了后续的实验。

# 4 实验
我们在三个benchmark上测试了GraphSAGE：1. 使用Web of Science citation dataset对不同的学术文章进行主题分类。2. 对属于不同社区的Reddit posts进行分类，3. 对多个PPI图进行蛋白质功能分类。在所有的实验中，我们在训练时没有见过的顶点上做预测，对PPI上对完全未见过的图做预测。

**Experimental set-up.** 为了在inductive benchmark上面将经验结果置于上下文中考虑，我们对比了四个baseline：随即分类器，基于特征的逻辑回归，raw features和DeepWalk embedding拼接的embedding。我们也比较了GraphSAGE的四个变体，分别使用不同的聚合函数（3.3部分）。因为，GraphSAGE的卷积变体是一种扩展形式，是Kipf et al. 半监督GCN的inductive version，我们称这个变体为GraphSAGE-GCN。我们测试了根据式1的损失函数训练的GraphSAGE变体，还有在cross-entropy上训练的监督变体。对于所有的GraphSAGE变体我们使用ReLU作为激活，$K = 2$，邻居采样大小$S\_1 = 25$，$S\_2 = 10$（详情见4.4节）。

对于Reddit和citation数据集，我们使用"online"来训练DeepWalk，如Perozzi et al. 提到的那样，我们在做预测前，跑一轮新的SGD来嵌入新的测试顶点（详情见后记）。在多图设定中，我们不能使用DeepWalk，因为通过DeepWalk在不同不相交的图上运行后生成的嵌入空间对其他来说可以是arbitrarily rotated（后记D）。

所有的模型都是用tf实现的，用Adam优化（除了DeepWalk，使用梯度下降效果更好）。我们设计的实验目标是1. 验证GraphSAGE比其他方法好。 2. 严格对比集中聚合架构，为了严格对比，所有的方法使用相同的实现，如minibatch迭代器，损失函数和邻居采样（如果可以的话）。此外，为了防止对比聚合器时非有意的"hyperparameter hacking"，我们检查了所有GraphSAGE变体的超参数集合（为每个变体根据他们在验证集上的表现选择最好的设定）。可能的超参数集合在早期的验证集上决定，这个验证集是citation和Reddit的子集，后续就丢掉了。后记包含了实现的细节。