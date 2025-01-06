---
categories:
- 论文阅读笔记
date: 2018-07-13 16:54:31+0000
description: ICLR 2018。图注意力网络，使用 self-attention 来构建 graph attentional layer，attention
  会考虑当前顶点所有的邻居对它的重要性，基于谱图理论的模型不能应用到其他不同结构的图上，而这个基于attention的方法能有效的解决这个问题。原文链接：[Graph
  Attention Networks](https://arxiv.org/abs/1710.10903)
draft: false
math: true
tags:
- deep learning
- Attention
- Graph
title: Graph Attention Networks
---
ICLR 2018。图注意力网络，使用 self-attention 来构建 graph attentional layer，attention 会考虑当前顶点所有的邻居对它的重要性，基于谱图理论的模型不能应用到其他不同结构的图上，而这个基于attention的方法能有效的解决这个问题。原文链接：[Graph Attention Networks](https://arxiv.org/abs/1710.10903)
<!--more-->
# 摘要

我们提出了图注意力网络(GAT)，新型图神经网络，利用自注意力层解决基于图卷积及其相似方法的缺点。通过堆叠这种层（层中的顶点会注意邻居的特征），我们可以给邻居中的顶点指定不同的权重，不需要任何一种耗时的矩阵操作（比如求逆）或依赖图结构的先验知识。我们同时解决了基于谱的图神经网络的几个关键挑战，并且使我们的模型很轻松的应用在 inductive 或 transductive 问题上。我们的 GAT 模型在4个 transductive 和 inductive 图数据集上达到且匹敌当前最先进的算法：*Cora*, *Citeseer*, *Pubmed citation networks*，*protein-protin interaction*。

# 1 INTRODUCTION

CNN 结构可以有效的重复使用卷积核，在网格型的数据上应用。然而很多问题都是基于图结构的。

早期的工作使用RNN来处理图结构中数据的表示。2005 年和 2009 年提出了 GNN（Graph Neural Networks）的概念，作为 RNN 的泛化，可以直接处理更一般的图结构，比如带环图、有向、无向图。GCN 包含了迭代的过程，迭代时顶点状态向前传播，后面使用一个神经网络来产生输出。Li et al., 2016使用了Cho et al., 2014提出的门控循环单元进行改进。

进一步的研究分为谱方法和非谱方法。

一方面，谱方法使用图的谱表示，成功应用到了顶点分类的问题上。Bruna et al., 2014在傅里叶域中定义了卷积操作，通过计算拉普拉斯矩阵的特征值分解，由于潜在的大量的计算，引出了后续的非谱方法。Henaff et al., 2015引入了带有 smooth coefficients 的谱滤波器，可以使他们在空间局部化。后来 Defferrard et al., 2016 提出了通过拉普拉斯矩阵的切比雪夫多项式展开的近似表达。最后，Kipf & Welling 2017 通过限制滤波器只考虑 1 阶邻居从而简化了之前的方法。然而，前面提到的所有的谱方法，学习到的卷积核参数都依赖于拉普拉斯特征值分解后的特征向量，也就是说依赖于图结构。因此，训练在一个指定图结构的模型不能应用到不同结构的图上。

另一方面，还有一些非谱方法 (Duvenaud et al., 2015; Atwood & Towsley, 2016; Hamilton et al., 2017)，这些方法直接在图上定义卷积操作，直接在空间上相近的邻居上应用卷积操作。这些方法的一个挑战是需要定义一个能处理不同数量邻居的卷积操作，并且保证 CNN 权重共享的性质。在某些情况下，这需要学习为每个 node degree 学习一个权重矩阵 (Duvenaud et al., 2015)，在对每个 input channel 和 neighborhood degree 训练权重时，使用转移矩阵的幂定义邻居 (Atwood & Towsley, 2016)，或是对有着固定数量顶点的邻居进行提取和归一化(Niepert et al., 2016)。Monti et al., 2016提出了混合模型 CNN，(MoNet)，这个空间方法提供了一个 CNN 在图上泛化的统一的模型。最近，Hamilton et al., 2017提出了 GraphSAGE，对每个顶点采样采出一个固定数量的邻居，然后使用一个指定的聚合操作聚集他们（比如取所有采样邻居的均值，或是将他们放进RNN后产生的结果）。这个方法在几个大规模的 inductive 数据集上获得了很惊人的效果。

注意力机制在很多基于序列的任务中已经成为了一个标准 (Bahdanau et al., 2015; Gehring et al., 2016)。注意力机制的一个好处是可以处理变长输入，专注于输入中最相关的部分来做决策。使用注意力机制计算一个序列的表示时，一般提到的是 *self-attention* 或 *intra-attention*。与 RNN 或卷积一起使用时，self-attention 在机器阅读(Cheng et al., 2016)和学习句子表示(Lin et al., 2017)这些任务上很有用。然而，Vaswani et al., 2017的研究表明，self-attention 不仅可以提升 RNN 和卷积的模型，在机器翻译任务上也是可以构建出性能最强的模型的。

受最近工作的启发，我们提出了基于 attention 的架构对图结构的顶点进行分类。思路是通过对顶点邻居的注意，计算图中每个顶点的表示，然后使用一个 self-attention 机制。注意力架构有几个有趣的性质：(1) 操作高效，因为它可以“顶点-邻居”对上并行计算；(2) 通过指定对邻居任意的权重，它可以在有着不同度的顶点上使用；(3) 模型可以直接应用在 inductive learning 任务上，包括模型必须要生成完整的未见过的图等任务。我们在4个 benchmark 上验证了我们的方法：*Cora*，*Citeseer*，*Pubmed citation networks*，*protein-protein interaction*，获得了比肩 state-of-the-art 的结果，展现了基于 attention 的模型在处理任意结构的图的可能性。

值得注意的是，如 Kipf & Welling 2017和Atwood & Towsley 2016，我们的工作可以重写为 MoNet(Monti et al., 2016)的一种特殊形式。除此以外，我们的分享神经网络跨边计算时是对关系网络公式的联想(Santoro et al., 2017)和VAIN(Hoshen, 2017)，在这两篇文章中，object 和 agent 间的关系被聚合成对，通过使用一种共享机制。相似地，我们的注意力模型可以与 Duan et al., 2017 和 Denil et al., 2017 的工作相连，他们使用一个邻居注意力操作来计算环境中不同 object 的注意力系数。其他相关的方法包括局部线性嵌入(LLE)(Roweis & Saul, 2000)和记忆网络(Weston et al., 2014)。LLE 在每个数据点选择了固定数量的邻居，为每个邻居学习了权重系数，以此将每个数据点重构为邻居的加权之和。之后的优化提取了顶点嵌入的特征。记忆网络与我们的工作也有关系，如果我们将一个顶点的邻居解释为记忆，通过注意它的值可以计算顶点特征，之后通过在同样的位置存储新特征进行更新。

# 2 GAT ARCHITECTURE

我们会在这部分描述如何创建 block layer 来构造任意的 graph attention networks，并且指明理论和实际上的优点以及相比于之前在神经图处理上的工作的缺点。

## 2.1 Graph Attentional Layer

首先描述单个 graph attentional layer，因为这种层会在整个 GAT 架构中使用。我们使用的 attention 和 Bahdanau et al., 2015 的工作相似。

输入是一组顶点特征，${\mathbf{h}} = \lbrace \vec{h}\_1, \vec{h}\_2, ..., \vec{h}\_N \rbrace , \vec{h}\_i \in \mathbb{R}^F$，其中 $N$ 是顶点数，$F$ 是每个顶点的特征数。这个层会生成一组新的顶点特征，${\mathbf{h}'} = \lbrace \vec{h}'\_1, \vec{h}'\_2, ..., \vec{h}'\_N\rbrace , \vec{h}'\_i \in \mathbb{R}^{F'}$，作为输出。

为了在将输入特征变换到高维特征时获得充足的表现力，至少需要一个可学习的线性变换。为了到达这个目的，每个顶点都会使用一个共享参数的线性变换，参数为 ${\mathbf{W}} \in \mathbb{R}^{F' \times F}$。然后在每个顶点上做一个 self-attention ——一个共享的attention机制 $a : \mathbb{R}^{F'} \times \mathbb{R}^{F'} \rightarrow \mathbb{R}$ 来计算注意力分数 *attention coefficients*：

$$\tag{1}
e\_{ij} = a(\mathbf{W} \vec{h}\_i, \mathbf{W} \vec{h}\_j)
$$

表示顶点 $j$ 的特征对顶点 $i$ 的重要性(*importance*)。在一般的公式中，模型可以使每个顶点都注意其他的每个顶点，扔掉所有的结构信息。我们使用 *mask attention* 使得图结构可以注入到注意力机制中——我们只对顶点 $j \in \mathcal{N\_i}$ 计算$e\_{ij}$，其中$\mathcal{N\_i}$ 是顶点 $i$ 在图中的一些邻居。在我们所有的实验中，这些是 $i$ 的一阶邻居（包括 $i$ ）。为了让系数在不同的顶点都可比，我们对所有的 $j$ 使用 softmax 进行了归一化：

$$\tag{2}
\alpha\_{ij} = \mathrm{softmax}\_j (e\_{ij}) = \frac{\exp{e\_{ij}}}{\sum\_{k \in \mathcal{N}\_i} \exp{e\_{ik}}}
$$

<div align="center">![Figure1](/images/graph-attention-networks/Fig1.JPG)</div>

在我们的实验中，注意力机制 $a$ 是一个单层的前向传播网络，参数为权重向量 $\vec{\text{a}} \in \mathbb{R}^{2F'}$，使用LeakyReLU作为非线性层（斜率$\alpha = 0.2$）。整个合并起来，注意力机制计算出的分数（如图1左侧所示）表示为：

$$\tag{3}
\alpha\_{ij} = \frac{ \exp{ ( \mathrm{LeakyReLU} ( \vec{\text{a}}^T [\mathbf{W} \vec{h}\_i \Vert \mathbf{W} \vec{h}\_j ] ))}}{\sum\_{k \in \mathcal{N\_i}} \exp{(\mathrm{LeakyReLU}(\vec{\text{a}}^T [\mathbf{W} \vec{h}\_i \Vert \mathbf{W} \vec{h}\_k]))}}
$$

其中 $·^T$ 表示转置，$\Vert$ 表示concatenation操作。

得到归一化的分数后，使用归一化的分数计算对应特征的线性组合，作为每个顶点最后的输出特征（最后可以加一个非线性层，$\sigma$）：

$$\tag{4}
\vec{h}'\_i = \sigma(\sum\_{j \in \mathcal{N}\_i} \alpha\_{ij} \mathbf{W} \vec{h}\_j)
$$

为了稳定 self-attention 的学习过程，我们发现使用 *multi-head attention* 来扩展我们的注意力机制是很有效的，就像 Vaswani et al., 2017。特别地，$K$ 个独立的 attention 机制执行 式4 这样的变换，然后他们的特征连(concatednated)在一起，就可以得到如下的输出：

$$\tag{5}
\vec{h}'\_i = \Vert^{K}\_{k=1} \sigma(\sum\_{j \in \mathcal{N}\_i} \alpha^k\_{ij} \mathbf{W}^k \vec{h}\_j)
$$

其中 $\Vert$ 表示concatenation，$\alpha^k\_{ij}$ 是通过第 $k$ 个注意力机制 $(a^k)$ 计算出的归一化的注意力分数，$\mathbf{W}^k$ 是对应的输入线性变换的权重矩阵。注意，在这里，最后的返回输出 $\mathbf{h}'$，每个顶点都会有 $KF'$ 个特征（不是 $F'$ ）。
特别地，如果我们在网络的最后一层使用 multi-head attention，concatenation 就不再可行了，我们会使用 *averaging*，并且延迟使用最后的非线性层（分类问题通常是 softmax 或 sigmoid ）：

$$
\vec{h}'\_i = \sigma(\frac{1}{K} \sum^K\_{k=1} \sum\_{j \in \mathcal{N}\_i} \alpha^k\_{ij} \mathbf{W}^k \vec{h}\_j)
$$

multi-head 图注意力层的聚合过程如图1右侧所示。

## 2.2 Comparisons to related work

2.1节描述的图注意力层直接解决了之前在图结构上使用神经网络建模的方法的几个问题：
· 计算高效：self-attention层的操作可以在所有的边上并行，输出特征的计算可以在所有顶点上并行。没有耗时的特征值分解。单个的GAT计算$F'$个特征的时间复杂度可以压缩至$O(\vert V \vert F F' + \vert E \vert F')$，$F$是输入的特征数，$\vert V \vert$和$\vert E \vert$是图中顶点数和边数。复杂度与Kipf & Welling, 2017的GCN差不多。尽管使用multi-head attention可以并行计算，但也使得参数和空间复杂度变成了$K$倍。
· 对比GCN，我们的模型允许对顶点的同一个邻居分配不同的重要度，使得模型能力上有一个飞跃。不仅如此，对学习到的attentional权重进行分析可以得到更好的解释性，就像机器翻译领域一样（比如Bahdanau et al., 2015的定性分析）。
· 注意力机制以一种共享的策略应用在图的所有的边上，因此它并不需要在之前就需要得到整个图结构或是所有的顶点的特征（很多之前的方法的缺陷）。因此这个方法有几个影响：
- 图不需要是无向的（如果边$j \rightarrow i$没有出现，我们可以直接抛弃掉$\alpha\_{ij}$的计算）
- 这个方法可以直接应用到*inductive learning*——包括在训练过程中在完全未见过的图上评估模型的任务上。最近发表的Hamilton et al., 2017的inductive方法为了保持计算过程的一致性，对每个顶点采样采了一个固定数量的邻居；这就使得这个方法在推断的时候不能考虑所有的邻居。此外，在使用一个基于LSTM的邻居聚合方式时（Hochreiter & Schmidhuber, 1997），这个方法在某些时候能达到最好的效果。这意味着存在邻居间存在一个一致的顶点序列顺序，作者通过将随机顺序的序列输入至LSTM来验证它的一致性。我们的方法不会受到这些问题中任意一个的影响——它在所有的邻居上运算（虽说会有计算上的开销，但扔能和GCN这样的速度差不多），并且假设任意顺序都可以。
· 如第一节提到的，GAT可以重写成MoNet(Monti et al., 2016)的一种特殊形式。更具体的来说，设pseudo-coordinate function为$u(x, y) = f(x) \Vert f(y)$，$f(x)$表示$x$的特征（可能是MLP变换后的结果），$\Vert$表示concatenation；权重函数为$w\_j(u) = \mathrm{softmax}(\mathrm{MLP}(u))$（softmax在一个顶点所有的邻居上计算）会使MoNet的patch operator和我们的很相似。尽管如此，需要注意到的是，对比之前MoNet的实例，我们的模型使用顶点特征计算相似性，而不是顶点的结构性质（这需要之前就已经直到图结构）。

我们可以做出一种使用稀疏矩阵操作的GAT层，将空间复杂度降低到顶点和边数的线性级别，使得GAT模型可以在更大的图数据集上运行。然而，我们使用的tensor操作框架只支持二阶tensor的稀疏矩阵乘法，限制了当前实现的版本的模型能力（尤其在有多个图的数据集上）。解决这个问题是未来的一个重要研究方向。在这些使用稀疏矩阵的场景下，在某些图结构下GPU的运算并不能比CPU快多少。另一个需要注意的地方是我们的模型的感受野的大小的上届取决于网络的深度（与GCN和其他模型相似）。像skip connections(He et al., 2016)这样的技术可以来近似的扩展模型的深度。最后，在所有边上的并行计算，尤其是分布式的计算可以设计很多冗余的计算，因为图中的邻居往往高度重叠。

# 3 Evaluation
我们与很多强力的模型进行了对比，在四个基于图的数据集上，达到了state-of-the-art的效果。这部分将总结一下我们的实验过程与结果，并对GAT提取特征表示做一个定性分析。
![Table1](/images/graph-attention-networks/Table1.JPG)

## 3.1 Datasets
**Transductive learning** 我们使用了三个标准的citation network benchmark数据集——Cora, Citeseer和Pubmed(Sen et al., 2008)——并按Yang et al., 2016做的transductive实验。这些数据集中，顶点表示文章，边（无向）表示引用。顶点特征表示文章的BOW特征。每个顶点有一个类标签。我们使用每类20个顶点用来训练，训练算法使用所有的顶点特征。模型的预测性能是在1000个测试顶点上进行评估的，我们使用了500个额外的顶点来验证意图（像Kipf & Welling 2017）。Cora数据集包含了2708个顶点，5429条边，7个类别，每个顶点1433个特征。Citeseer包含3327个顶点，4732条边，6类，每个顶点3703个特征。Pubmed数据集包含19717个顶点，44338条边，3类，每个顶点500个特征。

**Inductive learning** 我们充分利用protein-protein interaction(PPI)数据集，这个数据集包含了不同的人体组织（Zitnik & Leskovec, 2017）。数据集包含了20个图来训练，2个验证，2个测试。关键的是，测试的图包含了训练时完全未见过的图。为了构建图，我们使用Hamilton et al., 2017预处理后的数据。平均每个图的顶点数为2372个。每个顶点有50个特征，组成了positional gene sets，motif gene sets and immunological signatures。从基因本体获得的每个顶点集有121个标签，由Molecular Signatures Database(Subramanian et al., 2005)收集，一个顶点可以同时拥有多个标签。

这些数据集的概貌在表1中给出。

## 3.2 State-of-the-art methods
**Transductive learning** 对于transductive learning任务，我们对比了Kipf & Welling 2017的工作，以及其他的baseline。包括了label propagation(LP)(Zhu et al., 2003)，半监督嵌入(SemiEmb)(Weston et al., 2012)，manifold regulariization(ManiReg)(Belkin et al., 2006)，skip-gram based graph embeddings(DeepWalk)(Perozzi et al., 2014)，the iterative classification algorithm(ICA)(Lu & Getoor, 2003)和Planetoid(Yang et al., 2016)。我们也直接对比了GCN(Kipf & Welling 2017)，还有利用了高阶切比雪夫的图卷积模型(Defferrard et al., 2016)，还有Monti et al., 2016提出的MoNet。

**Inductive learning** 对于inductive learning任务，我们对比了Hamilton et al., 2017提出的四个不同的监督的GraphSAGE 这些方法提供了大量的聚合特征：GraphSAGE-GCN（对图卷积操作扩展inductive setting），GraphSAGE-mean（对特征向量的值取element-wise均值），GraphSAGE-LSTM（通过将邻居特征输入到LSTM进行聚合），GraphSAGE-pool（用一个共享的多层感知机对特征向量进行变换，然后使用element-wise取最大值）。其他的transductive方法要么在inductive中完全不合适，要么就认为顶点是逐渐加入到一个图中，使得他们不能在完全未见过的图上使用（如PPI数据集）。

此外，对于两种任务，我们提供了每个顶点共享的MLP分类器（完全没有整合图结构信息）的performance。

## 3.3 Experimental Setup
**Transductive learning** 我们使用一个两层的GAT模型。超参数在Cora上优化过后在Citeseer上复用。第一层包含$K = 8$个attention head，计算得到$F' = 8$个特征（总共64个特征），之后接一个指数线性单元（ELU）（Clevert et al., 2016）作为非线性单元。第二层用作分类：一个单个的attention head计算$C$个特征（其中$C$是类别的数量），之后用softmax激活。处理小训练集时，在模型上加正则化。在训练时，我们使用$L\_2$正则化，$\lambda = 0.0005$。除此以外，两个层的输入都使用了$p = 0.6$的dropout(Srivastava et al., 2014)，在*normalized attention coefficients*上也使用了（也就是在每轮训练时，每个顶点都被随机采样邻居）。如Monti et al., 2016观察到的一样，我们发现Pubmed的训练集大小(60个样本)需要微调：我们使用$K = 8$个attention head，加强了$L\_2$正则，$\lambda = 0.001$。除此以外，我们的结构都和Cora和Citeseer的一样。

**Inductive learning**
我们使用一个三层的GAT模型。前两层$K = 4$，计算$F' = 256$个特征（总共1024个特征），然后使用ELU。最后一层用于多类别分类：$K = 6$，每个计算121个特征，取平均后使用logistic sigmoid激活。训练集充分大所以不需要使用$L\_2$正则或dropout——但是我们使用了skip connections(He et al., 2016)在attentional layer间。训练时batch size设置为2个图。为了严格的衡量出使用注意力机制的效果（与GCN相比），我们也提供了*constant attention mechanism*，$a(x, y) = 1$，使用同样的架构——也就是每个邻居上都有相同的权重。

两个模型都使用了Glorot初始化(Glorot & Bengio, 2010)，使用Adam SGD(Kingma & Ba, 2014)优化cross-entropy，Pubmed上初始学习率是0.01，其他数据集是0.005。我们在cross-entropy loss和accuracy(transductive)或micro-F1(inductive)上都使用了early stopping策略，迭代次数为100轮。
代码：https://github.com/PetarV-/GAT

## 3.4 Results
![Table2](/images/graph-attention-networks/Table2.JPG)
对于transductive任务，我们提交了我们的方法100次的平均分类精度（还有标准差），也用了Kipf & Welling., 2017和Monti et al., 2016的metrics。特别地，对于基于切比雪夫方法(Defferrard et al., 2016)，我们提供了二阶和三阶最好的结果。为了公平的评估注意力机制的性能，我们还评估了一个计算出64个隐含特征的GCN模型，并且尝试了ReLU和ELU激活，记录了100轮后更好的那个结果（GCN-64*）（结果显示ReLU更好）。
![Table3](/images/graph-attention-networks/Table3.JPG)
对于inductive任务，我们计算了micro-averaged F1 score在两个从未见过的测试图上，平均了10次结果，也使用了Hamilton et al., 2017的metrics。特别地，因为我们的方法是监督的，我们对比了GraphSAGE。为了评估聚合所有的邻居的优点，我们还提供了我们通过修改架构（三层GraphSAGE-LSTM分别计算[512, 512, 726]个特征，128个特征用来聚合邻居）所能达到的GraphSAGE最好的结果（GraphSAGE*）。最后，为了公平的评估注意力机制对比GCN这样的聚合方法，我们记录了我们的constant attention GAT模型的10轮结果（Const-GAT）。
结果展示出我们的方法在四个数据集上都很好，和预期一致，如2.2节讨论的那样。具体来说，在Cora和Citeseer上我们的模型上升了1.5%和1.6%，推测应该是给邻居分配不同的权重起到了效果。值得注意的是在PPI数据集上：我们的GAT模型对于最好的GraphSAGE结果提升了20.5%，这意味着我们的模型可以应用到inductive上，通过观测所有的邻居，模型会有更强的预测能力。此外，针对Const-GAT也提升3.9%，再一次展现出给不同的邻居分配不同的权重的巨大提升。

学习到的特征表示的有效性可以定性分析——我们提供了t-SNE(Maaten & Hinton, 2008)的可视化——我们对在Cora上面预训练的GAT模型中第一层的输出做了变换（图2）。representation在二维空间中展示出了可辩别的簇。注意，这些簇对应了数据集的七个类别，验证了模型在Cora上对七类的判别能力。此外，我们可视化了归一化的attention系数（对所有的8个attention head取平均）的相对强度。像Bahdanau et al., 2015那样适当的解释这些系数需要更多的领域知识，我们会在未来的工作中研究。
![Figure2](/images/graph-attention-networks/Fig2.JPG)

# 4 Conclusions
我们展示了图注意力网络(GAT)，新的卷积风格神经网络，利用masked self-attentional层。图注意力网络计算高效（不需要耗时的矩阵操作，在图中的顶点上并行计算），处理不同数量的邻居时对邻居中的不同顶点赋予不同的重要度，不需要依赖整个图的结构信息——因此解决了之前提出的基于谱的方法的问题。我们的这个利用attention的模型在4个数据集针对transductive和inductive（特别是对完全未见过的图），对顶点分类成功地达到了state-of-the-art的performance。

未来在图注意力网络上有几点可能的改进与扩展，比如解决2.2节描述的处理大批数据时的实际问题。还有一个有趣的研究方向是利用attention机制对我们的模型进行一个深入的解释。此外，扩展我们的模型从顶点分类到图分类也是一个更具应用性的方向。最后，扩展我们的模型到整合边的信息（可能制视了顶点关系）可以处理更多的问题。