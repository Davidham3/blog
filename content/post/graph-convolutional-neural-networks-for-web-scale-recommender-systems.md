---
categories:
- 论文阅读笔记
date: 2018-06-17 21:24:48+0000
draft: false
math: true
tags:
- deep learning
- recommender system
- graph convolutional network
- large-scale learning
- Graph
title: Graph Convolutional Neural Networks for Web-Scale Recommender Systems
---
KDD 2018。使用图卷积对顶点进行表示，学习顶点的 embedding ，通过卷积将该顶点的邻居信息融入到向量中。原文链接：[Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/abs/1806.01973v1)。

<!--more-->

# ABSTRACT

最近在图数据上的深度神经网络在推荐系统上表现的很好。然而，把这些算法应用到数十亿的物品和数百亿的用户上仍然是个挑战。

我们提出了一种在 Pinterest 上的大规模深度推荐引擎，开发了一种高效的图卷积算法 PinSage，融合了随机游走和图卷积，来生成顶点（物品）的表示，同时整合了顶点信息和图结构。对比之前的 GCN 方法，我们研究的模型基于高效的随机游走来结构化卷积操作，而且还设计了一个新型的训练策略，这个策略依赖于 harder-and-harder 训练样本，来提高模型的鲁棒性和收敛能力。

我们的 PinSage 在 Pinterest 上面的75亿个样本上进行训练，图上有30亿个顶点表示 *pins* 和 *boards*，180亿条边。根据离线指标、用户研究和 A/B 测试，PinSage 生成了相比其他深度学习和基于图的方法更高质量的推荐结果。据我们所知，这是深度图表示目前规模最大的应用，并且为新一代基于图卷积结构的大规模推荐系统奠定了基础。

# 1 INTRODUCTION

深度学习方法在推荐系统中越来越重要，用来学习图像、文本、甚至是用户的有效的低维表示。使用深度学习学习到的表示可以用来补充、或是替换像协同过滤这样传统的推荐算法。这些表示很有用，因为他们可以在各种推荐任务中重复使用。举个例子，使用深度模型学习得到的物品的表示，可以用来做 “物品-物品” 推荐，也可以来按主题推荐（比如，歌单、或是 Feed流的内容）。

近些年可以看到这个领域的很多重要的发展，尤其是新的可以学习图结构的深度学习方法的发展，是一些推荐应用的基础（比如在用户-物品网络上或社交网络上推荐）。

在这些成功的深度学习框架中比较重要的是图卷积网络（GCN）。核心的原理是学习如何迭代地使用神经网络从局部图邻居中聚合特征信息（图1）。这里，一个简单的卷积操作从一步邻居中变换并聚合特征信息，并且通过堆叠多个这样的卷积，信息可以传播到图中很广的地方。不像纯基于内容的深度模型（如 RNN ），GCN 利用内容信息和图结构。基于 GCN 的模型的方法已经在无数推荐系统中形成了新的标准（参见[19]的综述）。然而，这些b enchmark 上面获得的提升，还没有被转换到真实环境的应用中去。

主要挑战是要将训练和基于 GCN 的顶点表示在数十亿的顶点和数百亿的边的图中进行。扩展 GCN 很困难，因为很多在大数据环境中，很多基于这些 GCN 设计的假设都不成立了。比如，所有的基于 GCN 的推荐系统需要在训练时使用图的拉普拉斯矩阵，但是当顶点数很大的时候，这就不现实了，因为算不出来。

# 3 METHOD

在这部分，我们将描述 PinSage 的结构和训练的技术细节，也会讲一下使用训练好的 PinSage 模型来高效地生成 embedding 的MapReduce pipeline。

![Figure1](/images/graph-convolutional-neural-networks-for-web-scale-recommender-systems/Fig1.PNG)

我们方法的计算关键在于局部图卷积的表示(notion)。我们使用多个卷积模块来聚合一个顶点局部的邻域特征信息（图1），来生成这个顶点的 embedding（比如一个物品）。每个模块学习如何从一个小的图邻域中聚合信息，并且通过堆叠多个这样的模块，我们的方法可以获得局部网络的拓扑结构信息。更重要的是，这些局部卷积模块的参数对所有的顶点来说是共享的，这使得我们的方法的参数的计算复杂度与输入的图的大小无关。

# 3.1 Problem Setup

Pinterest 是一个内容挖掘应用，在这里用户与 *pins* 进行交互，这些 *pins* 是在线内容的可见标签（比如用户做饭时的食谱，或者他们想买的衣服）。用户用 *boards* 将 *pins* 组织起来，*boards* 里面包含了 *pins* 组成的集合，这些 *pins* 在用户看来是主题相关的。Pinterest 组成的图包含了 20 亿的 *pins*，10 亿的 *boards*，超过 180 亿的边（也就是 *pins* 对应 *boards* 的关系）。

我们的任务是生成可以用于推荐的高质量的 embedding 或 *pins* 的表示（比如，使用最近邻来查找 *pin* 的推荐，或是使用下游的再评分系统进行推荐）。为了学习这些 embedding，我们对 Pinterest 环境进行建模，得到一个二部图，顶点分为两个不相交的集合，$\mathcal{I}$ 表示 *pins*，$\mathcal{C}$ 表示 *boards*。当然，我们的方法是可以泛化到其他方面的，比如 $\mathcal{I}$ 看作是物品，$\mathcal{C}$ 看作是用户定义的环境或收藏品集合等。

再来说说图结构，我们假设 *pins/items* $u \in \mathcal{I}$ 与特征 $x\_u \in \mathbb{R}^d$ 相关。通常来说，这些特征可能是物品的元数据或上下文信息，在 Pinterest 的例子中，*pins* 是和富文本与图片特征相关的。我们的目标是利用这些输入特征，也利用二部图的图结构性质来生成高质量的 embedding。这些 embedding 可以用于推荐系统，通过最近邻查找来生成推荐，或是作为用评分来推荐的机器学习系统的特征。

为了符号的简洁，我们使用 $\mathcal{V} = \mathcal{I} \cup \mathcal{C}$ 来表示图中的顶点集，没有特殊需要不区分 *pin* 和 *board* 顶点，一律使用 *node* 来表示顶点。

## 3.2 Model Architecture

我们使用局部卷积模块对顶点生成 embeddings。首先输入顶点的特征，然后学习神经网络，神经网络会变换并聚合整个图上的特征来计算顶点的 embeddings（图1）。

**Forward propagation algorithm.** 考虑对顶点 $u$ 生成 embedding $z\_u$ 的任务，需要依赖顶点的输入特征和这个顶点周围的图结构。

!["Algorithm 1"](/images/graph-convolutional-neural-networks-for-web-scale-recommender-systems/algo1.PNG)

我们的 PinSage 算法是一个局部卷积操作，我们可以通过这个局部卷积操作学到如何从 $u$ 的邻居聚合信息（图1）。这个步骤在算法1 CONVOLVE 中有所描述。从本质上来说，我们通过一个全连接神经网络对 $\forall{v} \in \mathcal{N}(u)$，也就是 $u$ 的邻居的表示 $z\_v$ 进行了变换，之后在结果向量集合上用一个聚合/池化函数（例如：一个 element-wise mean 或是加权求和，表示为 $\gamma$）（Line 1）。这个聚合步骤生成了一个 $u$ 的邻居$\mathcal{N}(u)$ 的表示 $n\_u$。之后我们将这个聚集邻居向量 $n\_u$ 和 $u$ 的当前表示向量进行拼接后，输入到一个全连接神经网络做变换（Line 2）。通过实验我们发现使用拼接操作会获得比平均操作[21]好很多的结果。除此以外，第三行的 normalization 使训练更稳定，而且对近似最近邻搜索来说归一化的 embeddings 更高效（Section 3.5）。算法的输出是集成了 $u$ 自身和他的局部邻域信息的表示。

**Importance-based neighborhoods.** 我们方法中的一个重要创新是如何定义的顶点邻居 $\mathcal{N}(u)$，也就是我们在算法1中是如何选择卷积的邻居集合。尽管之前的 GCN 方法简单地检验了 k-hop 邻居，在 PinSage 中我们定义了基于重要性的邻域，顶点 $u$ 的邻居定义为 $T$ 个顶点，这 $T$ 个顶点对 $u$ 是最有影响力的。具体来说，我们模拟了从顶点 $u$ 开始的随机游走，并且计算了通过随机游走[[14]](https://arxiv.org/abs/1711.07601)对顶点的访问次数的 $L\_1$ 归一化值。$u$ 的邻居因此定义为针对顶点 $u$ 来说 $T$ 个最高的归一化的访问数量的顶点。

这个基于重要性的邻域定义的优点有两点。第一点是选择一个固定数量的邻居顶点来聚集可以在训练过程中控制内存开销[18]。第二，在算法1中聚集邻居的向量表示时可以考虑邻居的重要性。特别地，我们在算法1中实现的 $\gamma$ 是一个加权求均值的操作，权重就是 $L\_1$ 归一化访问次数。我们将这个新的方法称为重要度池化(*importance pooling*)。

**Stacking convolutions.** 每次使用算法1的 CONVOLVE 操作都会得到一个顶点的新的表示，我们可以在每个顶点上堆叠卷积来获得更多表示顶点 $u$ 的局部邻域结构的信息。特别地，我们使用多层卷积，其中对第 $k$ 层卷积的输入依赖于 $k-1$ 层的输出（图1），最初的表示（"layer 0"）等价于顶点的输入特征。需要注意的是，算法1中的模型参数（$Q$, $q$, $W$ 和 $w$）在顶点间是共享的，但层与层之间不共享。

![](/images/graph-convolutional-neural-networks-for-web-scale-recommender-systems/algo2.PNG)

算法2详细描述了如何堆叠卷积操作，针对一个 minibatch 的顶点 $\mathcal{M}$ 生成 embeddings。首先计算每个顶点的邻居，然后使用 $K$ 个卷积迭代来生成目标顶点的 K 层表示。最后一层卷积层的输出之后会输入到一个全连接神经网络来生成最后的 embedding $z\_u$，$\forall{u} \in \mathcal{M}$。

模型需要学习的参数有：每个卷积层的权重和偏置（$Q^{(k)}$，$q^{(k)}$，$W^{(k)}$，$w^{(k)}$，$\forall{k} \in \lbrace 1,...,K\rbrace $），还有最后的全连接网络中的参数 $G\_1$，$G\_2$，$g$。算法1的第一行的每层输出的维度（也就是 $Q$ 的列空间的维度）设为 $m$。为了简单起见，我们将所有卷积层（算法1的第三行的输出）的输出都设为同一个数，表示为 $d$。模型最后的输出（算法2第18行之后）也设为 $d$。

## 3.3 Model Training

我们使用 max-margin ranking loss 来训练 PinSage。在这步，假设我们有了一组标记的物品对 $\mathcal{L}$，$(q,i) \in \mathcal{L}$ 认为是相关的，也就是当查询 $q$ 时，物品 $i$ 是一个好的推荐候选项。训练阶段的目标是优化 PinSage 的参数，使得物品对 $(q,i) \in \mathcal{L}$ 的 embedding 在标记集合中尽可能的接近。

我们先来看看 margin-based loss function。首先我们来看看我们使用的可以高效地计算并且使 PinSage 快速收敛的一些技术，这些技术可以让我们训练包含数十亿级别的顶点的图，以及数十亿训练样本。最后，我们描述我们的 curriculum-training scheme，这个方法可以全方位的提升我们的推荐质量。

**Loss function.** 为了训练模型的参数，我们使用了一个基于最大边界的损失函数。基本的思想是我们希望最大化正例之间的内积，也就是说，查询物品的 embedding 和对应的相关物品的 embedding 之间的内积。与此同时我们还想确保负例之间的内积，也就是查询物品的 embedding 和那些不相关物品 embedding 之间的内积要小于通过提前定义好的边界划分出的正例的内积。对于单个顶点对 embeddings $(z\_q, z\_i):(q, i) \in \mathcal{L}$ 的损失函数是：

$$
J\_{\mathcal{G}}(z\_qz\_i) = \mathbb{E}\_{n\_k \thicksim p\_n(q)}\max\lbrace 0, z\_q \cdot z\_{n\_k}-z\_q \cdot z\_i + \Delta\rbrace 
$$

其中，$P\_n(q)$ 表示物品 $q$ 的负样本分布，$\Delta$ 表示 margin 超参数。一会儿会讲负样本采样。

**Multi-GPU training with large minibatches.** 为了在训练中充分利用单台机器的多个 GPU，我们以一种 multi-tower 的方法运行前向和反向传播。我们首先将每个 minibatch（图1底部）分成相等大小的部分。每个 GPU 获得 minibatch 的一部分，使用同一组参数来运算。在反向传播之后，所有 GPU 上的针对每个参数的梯度进行汇集，然后使用一个同步的 SGD。由于训练需要极大数量的样本，我们在运行时使用了很大的 batch size，范围从 512 到 4096。

我们使用与 Goyal et al.[16] 提出的相似的技术来确保快速收敛，而且在处理大 batch size 时训练的稳定和泛化精度。我们在第一轮训练的时候根据线性缩放原则使用一个 gradual warmup procedure，使学习率从小增大到一个峰值。之后学习率以指数级减小。

**Producer-consumer minibatch construction.** 在训练的过程中，由于邻接表和特征矩阵有数十亿的顶点，所以放在了 CPU 内存中。然而，在训练 PinSage 的 CONVOLVE 步骤时，每个 GPU 需要处理邻居和顶点邻居的特征信息。从 GPU 访问 CPU 内存中的数据时会有很大的开销。为了解决这个问题，我们使用了一个 *re-indexing* 的方法创建包含了顶点和他们的邻居的子图 $G' = (V', E')$，在当前的 minibatch 中会被加入到计算中。只包含当前 minibatch 计算的顶点特征信息的小的特征矩阵会被抽取出来，顺序与 $G'$ 中顶点的 index 一致。$G'$ 的邻接表和小的特征矩阵会在每个 minibatch 迭代时输入到 GPU 中，这样就没有了 GPU 和 CPU 间的通信开销了，极大的提高了 GPU 的利用率。

训练过程改变了 CPU 和 GPU 的使用方式。模型计算是在 GPU，特征抽取、re-indexing、负样本采样是在 CPU 上运算的。使用 multi-tower 训练的 GPU 并行和 CPU 计算使用了 OpenMP[25]，我们设计了一个生产者消费者模式在当前迭代中使用 GPU 计算，在下一轮使用 CPU 计算，两者并行进行。差不多减少了一半的时间。

**Sampling negative items.** 负样本采样在我们的损失函数中作为 edge likelihood[23] 的归一化系数的近似值。为了提升 batch size 较大时的训练效率，我们采样了 500 个负样本作为一组，每个 minibatch 的训练样本共同使用这一组。相比于对每个顶点在训练时都进行负样本采样，这极大地减少了每次训练时需要计算的 embeddings 的数量。从实验上来看，我们发现这两种方法在表现上没什么特别大的差异。

在最简单的情况中，我们从整个样本集中使用均匀分布的抽样方式。然而，确保正例($(q, i)$)的内积大于 $q$ 和 500 个负样本中每个样本的内积是非常简单的，而且这样做不能提供给系统足够学习的分辨率。我们的推荐算法应该能从 200 亿个商品中找到对于物品 $q$ 来说最相关的 1000 个物品。换句话说，我们的模型应该能从超过 2 千万的物品中区分/辨别出 1 件物品。但是通过随机采样的 500 件物品，模型的分辨率只是 $\frac{1}{500}$。因此，如果我们从 200 亿物品中随机抽取 500 个物品，这些物品中的任意一个于当前这件查询的物品相关的几率都很小。因此，模型通过训练不能获得好的参数，同时也不能对相关的物品进行区分的概率很大。为了解决上述问题，对于每个正训练样本（物品对$(q, i)$），我们加入了"hard"负例，也就是那些与查询物品 $q$ 有某种关联的物品，但是又不与物品 $i$ 有关联。我们称这些样本为"hard negative items"。通过在图中根据他们对查询物品 $q$ 的个性化 PageRank 分数来生成[14]。排名在 2000-5000 的物品会被随机采样为 hard negative items。如图2所示，hard negative examples 相比于随机采样的负样本更相似于查询物品，因此对模型来说挑战是排名，迫使模型学会在一个好的粒度上分辨物品。

![](/images/graph-convolutional-neural-networks-for-web-scale-recommender-systems/Fig2.PNG)

使用 hard negative items 会让能使模型收敛的训练轮数翻倍。为了帮助模型收敛，我们使用了 curriculum training scheme[4]。在训练的第一轮，不适用 hard negative items，这样算法可以快速地找到 loss 相对较小的参数空间。之后我们在后续的训练中加入了 hard negative items，专注于让模型学习如何从弱关系中区分高度关联的pins。在第 $n$ 轮，我们对每个物品的负样本集中加入了 $n-1$ 个 hard negative items。

## 3.4 Node Embeddings via MapReduce 

在模型训练结束后，对于所有的物品（包括那些在训练中未见过的物品）直接用训练好的模型生成 embeddings 还是有挑战的。使用算法2对顶点直接计算 embedding 的方法会导致重复计算，这是由顶点的K-hop 邻居导致的。如图1所示，很多顶点在针对不同的目标顶点生成 embedding 的时候被很多层重复计算多次。为了确保推算的有效性，我们使用了一种 MapReduce 架构来避免在使用模型进行推算的时候的重复计算问题。

我们发现顶点的 embedding 在推算的时候会很好的将其自身带入到 MR 计算模型中。图3详细地表述了 pin-to-board Pinterest 二部图上的数据流，我们假设输入（"layer-0"）顶点是 pins/items（layer-1 顶点是 boards/contexts）。MR pipeline 有两个关键的组成部分：
1. 一个 MapReduce 任务将所有的 pins 投影到一个低维隐空间中，在这个空间中会进行聚合操作（算法1，第一行）
2. 另一个 MR 任务是将结果的 pins 表示和他们出现在的 boards 的 id 进行连接，然后通过 board 的邻居特征的池化来计算 board 的 embedding。

注意，我们的方法避免了冗余的计算，对于每个顶点的隐向量只计算一次。在获得 boards 的 embedding 之后，我们使用两个以上的 MR 任务，用同样的方法计算第二层 pins 的 embedding，这个步骤也是可以迭代的（直到 K 个卷积层）。

### 3.5 Efficient nearest-neighbor lookups 
由 PinSage 生成的 embeddings 可以用在很多下游推荐任务上，在很多场景中我们可以直接使用这些 embeddings 来做推荐，通过在学习到的嵌入空间中使用最近邻查找。也就是，给定一个查询物品 $q$，我们使用 K-近邻的方式来查找查询物品 embedding 的 K 个邻居的嵌入。通过局部敏感哈希[2]的近似 K 近邻算法很高效。在哈希函数计算出后，查找物品可以通过一个基于 Weak AND 操作[5]的两阶段查询实现。PinSage 模型是离线计算的并且所有节点的表示通过 MR 计算后存放到数据库中，高效的最近邻查找方法可以使系统在线提供推荐结果。