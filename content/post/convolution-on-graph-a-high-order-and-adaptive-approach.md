---
categories:
- 论文阅读笔记
date: 2018-07-16 11:01:00+0000
description: '重新定义了卷积的定义，利用$k$阶邻接矩阵，定义考虑$k$阶邻居的卷积，利用邻接矩阵和特征矩阵构建能同时考虑顶点特征和图结构信息的卷积核。在预测顶点、预测图、生成图三个任务上验证了模型的效果。原文链接：[Graph
  Convolution: A High-Order and Adaptive Approach](https://arxiv.org/abs/1706.09916)'
draft: false
math: true
tags:
- deep learning
- Graph
- graph convolutional network
title: 'Convolution on Graph: A High-Order and Adaptive Approach'
---
重新定义了卷积的定义，利用$k$阶邻接矩阵，定义考虑$k$阶邻居的卷积，利用邻接矩阵和特征矩阵构建能同时考虑顶点特征和图结构信息的卷积核。在预测顶点、预测图、生成图三个任务上验证了模型的效果。原文链接：[Graph Convolution: A High-Order and Adaptive Approach](https://arxiv.org/abs/1706.09916)
<!--more-->
# 摘要
我们提出了两个新的模块为图结构数据而设计：k阶卷积器和自适应滤波模块。重要的是，我们的框架(HA-GCN)是一种通用框架，可以在顶点和图上适应多种应用，还有图生成模型。我们的实验效果很好。在顶点分类和分子属性预测上超越了state-of-the-art。生成了超过32%的真实分子，在材料设计和药物筛选上很有效。

# 引言
图卷积网络通常应用在两个学习任务上：
·以顶点为中心：预测任务与顶点相关。GCN对图中的每个顶点输出一个特征向量，有效地反映顶点属性和邻居结构。举个例子，社交网络中，向量用于顶点分类和链路预测。有时也和顶点表示学习有关。
·以图为中心：预测任务和图相关。举个例子，化学领域中，分子可以被看作是一个图，原子作顶点，化学键是边。根据分子的物理和化学性质，构建图卷积网络对分子进行有意义地编码。这些任务对很多生活中的应用如材料设计、药物筛选很重要。在这种情况下，图卷积通常对图进行编码，使用编码进行图的预测。

和我们的工作最相关的是Kipf & Welling 2016a，他们的卷积只考虑了一阶邻居。我们的高阶操作器有着到达k阶邻居的高效设计。此外，我们还引入了自适应模块动态地基于局部图的连接和顶点属性调整权重。对比Li et al., 2015，他们将LSTM引入图中，我们的自适应模块可以解释成Xu et al., 2015提出的注意力机制。不像前人设计的模型要么是以顶点为重，要么是以图为中心，我们的HA-GCN框架是个通用的模型。除此以外，我们用HA-GCN构建了针对分子生成的图生成模型，比state-of-the-art的效果提高了很多。

贡献有两点：
·引入两个模块，构建新的图卷积网络架构HA-GCN
·提出了可以应用到以顶点为中心、图为中心的通用框架和图生成模型。在所有的任务上获得了state-of-the-art的效果。

# Preliminaries
**The Graph Model**
一个图$\mathcal{G}$表示为一个对$(V, E)$，$V = \lbrace v\_1, ..., v\_n \rbrace $是顶点集，$E \in V \times V$是边集。我们不区分有向和无向，因为我们的模型都能做。每个图可以表示为一个$n \times n$的邻接矩阵$A$，如果$v\_i$到$v\_j$有边，$A\_{i,j} = 1$否则为$0$。基于邻接矩阵，我们可以得到距离函数$d(v\_i, v\_j)$表示$v\_i$到$v\_j$的距离（最短距离）。此外，我们认为每个顶点$v\_i$与一个特征向量$X\_i \in \mathcal{R}^m$相关，我们使用$X = (X^T\_1, X^T\_2, ..., X^T\_n) \in \mathcal{R}^{n \times m}$表示特征矩阵。

**Graph Convolutional Networks (GCNs)**
首先，一个顶点$v\_j$在图$\mathcal{G}$上的卷积操作可以表示为：
$$
L\_{conv}(j) = \sum\_{i \in \mathcal{N}\_j} w\_{ij} X\_i + b\_j
$$
其中$X\_i \in R\_m$是顶点$v\_i$的输入特征，$b\_j$是偏置，$w\_{ij}$是权重，对$j$是非平稳且变化的。集合$\mathcal{N}\_j$表示scope of convolution。对于传统的应用，CNN通常设计成低维度的网格且对每个顶点有着相同的连接。举个例子，图像可以看作二维网格，图$\mathcal{G}$通过邻接的像素组成。$\mathcal{N}\_j$可以简单的定义为一个围绕在像素$j$的固定大小的block或window。

在更一般的图上，可以将$\mathcal{N}\_j$定义为顶点$v\_j$的邻接顶点的集合。比如，在Duvenaud et al. 2015的工作中，fingerprint(FP)卷积操作器的核心是计算邻居的均值，也就是对于所有的$(i, j)$，$w\_{ij} = 1$。通过邻接矩阵$A$，我们可以将这个操作写为
$$\tag{1}
L\_{FP} = AX.
$$

$A$和特征矩阵$X$的乘积得到一个所有邻居顶点特征的均值。Kipf & Welling 2016a提出的node-GCN使用线性组合与非线性变换得到的均值为：
$$\tag{2}
L\_{node-GCN} = \sigma(AXW).
$$

权重矩阵$W$，函数$\sigma(\cdot)$分别是特征$X$上的线性组合与非线性变换。

Bruna et al., 2013与Defferrard et al., 2016提采用了不同的方式在图的拉普拉斯矩阵的谱上做了卷积。$H$为拉普拉斯矩阵，正交分解为$H = U \Lambda U^T$($U$是正交矩阵，$\Lambda$是对角矩阵)。与其在式2中加入权重矩阵，谱卷积考虑的是$H$上的一个参数化的卷积操作；
$$\tag{3}
L\_{spectral} = U g\_\theta U^T X.
$$

这里$g\_\theta(\cdot)$是一个多项式函数，element-wisely应用在对角矩阵$\Lambda$上。

在讨论谱图卷积的优点时，作者提到$k$阶多项式多项式$g\_\theta(\cdot)$就是图的$k$阶局部，也就是说卷积会到达$k$阶邻居。对比式1和式2一阶邻居的均值，这能使信息在图上快速的传播。然而，考虑到$U \Lambda^n U^T = A^n$，多项式$g\_\theta(\cdot)$的选择并没有给$k$阶邻居一个明确的卷积操作，因为在卷积中不是所有的邻居都是占相同的份量。这使得我们提出了我们的高阶卷积操作。这些卷积的其他问题是，他们在图中是不变的。因此几乎不能捕获到使用卷积时不同地方的不同。这使得我们提出了自适应模块，成功的考虑了局部特征和图结构。（这里看的云里雾里的。。。）

# High-Order and Adaptive Graph Convolutional Network (HA-GCN)
**K-th Order Graph Convolution**
定义顶点$v\_j$的$k$阶邻居：$\mathcal{N}\_j = \lbrace  v\_i \in V \mid d(v\_i, v\_j) \leq k \rbrace $。可以通过对邻接矩阵$A$连乘得到$k$阶邻居。
**Proposition 1.** $A$是图$\mathcal{G}$的邻接矩阵，$A^k$的第$i$行第$j$列表示的是顶点$i$到顶点$j$的$k$步路径的个数。
我们定义$k$阶卷积如下：
$$\tag{4}
\tilde{L}^{(k)}\_{gconv} = (W\_k \circ \tilde{A}^k)X + B\_k,
$$

其中
$$\tag{5}
\tilde{A}^k = \min\lbrace A^k + I, 1\rbrace .
$$

其中$\circ$和$\min$分别表示element-wise矩阵乘法和最小值。$W\_k \in \mathcal{R}^{n \times n}$是权重矩阵，$B\_k \in \mathcal{R}^{n \times m}$是偏置矩阵。$\tilde{A}^k$是通过将$A^k + I$砍到1获得的。在$A^k$上增加单位阵是为了让图上的每个顶点都有自连接。砍到1是因为如果$A^k$有大于1的元素，砍到1确实会得到k阶邻居的卷积。卷积的输入$\tilde{L}^{(k)}\_{gconv}$是邻接矩阵$A \in \lbrace 0, 1\rbrace ^{n \times n}$和特征矩阵$X \in \mathcal{r}^{n \times m}$。输出的维度和$X$一样。如同名字所示，卷积操作$\tilde{L}^{(k)}\_{gconv}$取一个顶点的$k$阶邻居的特征向量作为输入，输出他们的加权平均。

式4的操作优雅地实现了我们在图上$k$阶邻居的idea，和传统的卷积一样，是kernel size为$k$的卷积。一方面，它可以看作是式2从1阶邻居到高阶的高效的泛化。另一方面，卷积器与式3的谱图卷积关系紧密，因为谱图中的$k$阶多项式也可以被看作是一种范围为$k$阶邻居$\mathcal{N}\_j$的操作。

**Adaptive Filtering Module**
基于式4，我们引入图卷积的自适应滤波器模块。它根据一个顶点的特征以及邻居的连接过滤卷积的权重。以化学中分子的图为例，在预测分子性质时，benzene rings比alkyl chains更重要。没有自适应模块，图卷积在空间上不变，而且不能按预期的那样工作。自适应滤波器的引入使得网络自适应地找到卷积目标并且更好的捕获局部的不一致。

自适应滤波器的想法来源于注意力机制Xu et al., 2015，他们在生成输出序列中对应的词时，自适应地的了有趣的像素。也可以看作是一种门的变体，这个门有选择的让信息通过LSTM。从技术上来讲，我们的自适应滤波器是一个在权重矩阵$W\_k$上的非线性操作器$g$：
$$\tag{6}
\tilde{W\_k} = g \circ W\_k,
$$

其中$\circ$表示element-wise矩阵乘法。事实上，操作器$g$是由$\tilde{A}^k$和$X$共同决定的，反映了顶点特征与图的连接，
$$
g = f\_{adp}(\tilde{A}^k, X).
$$

我们考虑了函数$f\_{adp}$的两个选项：
$$\tag{7}
f\_{adp/prod} = \mathrm{sigmoid}(\tilde{A}^kXQ)
$$
和
$$\tag{8}
f\_{adp/lin} = \mathrm{sigmoid}(Q \cdot [\tilde{A}^k, X]).
$$

这里，$[\cdot, \cdot]$表示矩阵拼接。第一个操作器通过$A$和$X$的内积考虑了顶点特征和图连接的的交互，第二个通过线性变换也实现了这个目的。事实上，我们发现线性自适应滤波器(8)比(7)在大多数任务上表现的更好。因此，我们在实验部分会采用并且记录线性的表现。自适应滤波器为了顶点的权重选择而设计出来，因此我们用了一个sigmoid非线性激活使它二值化。参数矩阵$Q$会让$f\_{adp}$的输出的维度与矩阵$A$对齐。不像当前已有的动态滤波器只从顶点或边的特征中生成权重，我们的自适应滤波器模块通过同时考虑顶点特征与图的连通性有着更全面的考虑。

**The Framework of HA-GCN**
通过在高阶卷积式4中加入自适应模块式6，我们得到HA的定义为：
$$
\tilde{L}^{(k)}\_{HA} = (\tilde{W}\_k \circ \tilde{A}^k)X = B\_k
$$

![Fig1](/images/convolution-on-graph-a-high-order-and-adaptive-approach/Fig1.JPG)
图1给了卷积器和HA-GCN框架的可视化。图1a展示了对于一个顶点，$k = 2$时的$\tilde{L}^{(k)}\_{HA}$：自适应滤波器$g$的底层加到权重矩阵$W\_1$和$W\_2$上，得到了自适应权重$\tilde{W}\_1$和$\tilde{W}\_2$（橙色和绿色线）；第二层把自适应权重和对应的邻接矩阵拼到一起为了卷积使用。图1b强调了卷积是在图上每个顶点都做的，并且是一层一层的。需要注意的是高阶卷积器和自适应权重可以和其他的神经网络架构/操作一起使用，比如全连接层，池化层，非线性变换。我们将我们的HA操作的图卷积层命名为HA-GCN。

在所有的卷积层之后，我们将那些来自不同阶的卷积层的输出特征拼接起来：
$$
L\_{HA} = [\tilde{L}^{(1)}\_{HA}, ..., \tilde{L}^{(K)}\_{HA}].
$$

HA-GCN的架构以特征矩阵$X \in \mathcal{R}^{n \times m}$（$n$是图的顶点数，$m$是顶点特征的维数）作为输入，输出一个维度为$n \times (mK)$的矩阵，特征的维数会乘以$K$倍。现在，我们将详细描述如何将HA-GCN应用到各种各样的问题上。

**以顶点为中心的预测：**HA-GCN的卷积之后，每个顶点和一个特征向量有关。特征向量可以用于顶点分类或回归。与网络表示学习也密切相关。使用以顶点为中心的设定，意味着我们可以给每个顶点学习出一个向量，向量反映了那个顶点周围的图的局部结构。我们的HA-GCN也给图中的每个顶点输出了一个向量。在这个场景下，HA-GCN可以看成是一个监督的图表示学习框架。

**以图为中心的预测：**为了解决不同尺度的图，输入的邻接矩阵和特征矩阵在底部和右侧加入了为0的padding。以顶点为中心和以图为中心的一个小不同是：以顶点为中心的任务中，一部分有标签/值的顶点作为训练，其他的作验证和输出，而以图为中心的任务，数据集是一组图（可能尺度不同），分为训练/验证/测试集。HA-GCN在这两种情况都能工作，HA卷积层中的参数是$(n^2)$，$n$是图的size（或是那些图中最大的size）。HA-GCN在顶点为中心的任务中比在图为中心的任务更容易过拟合，我们会在后面的实验部分描述这个问题。

**图生成模型：**从一组图$\bar{\mathcal{G}} = \lbrace \mathcal{G}\_1, ..., \mathcal{G}\_N \rbrace $中学习一个概率模型，通过这个模型我可以生成之前没见过但是和$\bar{\mathcal{G}}$中相似的图。通过variational auto-encoder(Kingma and Welling 2013)和adversarial auto-encoder(Makhzani et al., 2015)，图卷积网络可以适用于生成模型的任务甚至是判别模型。

一个自编码器总是由两部分组成：一个编码器和一个解码器。编码器将输入数据$X \in \mathcal{X}$映射到一个编码向量$Y \in \mathcal{Y}$，解码器将$\mathcal{Y}$映射回$\mathcal{X}$。我们称编码空间$\mathcal{Y}$为隐藏空间。为了使他成为一个生成模型，我们通常假设隐藏空间中有一个概率分布（比如高斯分布）。图生成模型能让我们生成分子的连续表达，通过搜索隐藏空间生成新的化学结构，可以用来指导材料设计和药物筛选。

# 实验
**Node-centric learning**
citation graphs上的监督文档分类，每个图包含文档的bag-of-words特征向量，还有文档之间的引用连接。我们把这个网络当成无向图，构建一个二值对称邻接矩阵$A$。每个文档有一个类标，目标是从文档的特征和引用的图对文档标签预测。统计数据(Sen et al., 2008)

Dataset|Nodes|Edges|Classes|Features
:-:|:-:|:-:|:-:|:-:
Citeseer|3327|4732|6|3703
Cora|2708|5429|7|1433
Pubmed|19717|44338|210|5414

**训练和架构：**我们使用和Kipf & Welling同样的GCN网络结构，除了将他们的一阶图卷积层换成了我们的HA层。我们使用$gcn\_{1,...,k}$表示$1$阶到$k$阶的图卷积层。$\mathrm{fc}k$表示有$k$个隐藏单元的全连接层。

Name|Architectures
:-:|:-:
GCN|gcn_{1}-fc128-gcn_{1}-fc1-softmax
gcn_{1, 2}|gcn{1, 2}-fc128-gcn{1, 2}-fc1-softmax
adp_gcn_{1, 2}|adp_gcn{1, 2}-fc128-adp_gcn{1, 2}-fc1-softmax

为了比较不同模型的性能，我们将数据集随机划分成训练/验证/测试集，比例为$7:1.5:1.5$，记录测试集的预测精度，表1。超参数：dropout rate $0.7$，L2 regularization $0.5 \cdot 10^{-8}$，hidden units $128$。从顶点表示学习角度看，前三个是半监督的模型，后四个是半监督的模型。这也解释了为什么后面的模型效果更好。我们的二姐邻居HA图卷积在精度上提升了2%。自适应模块没能继续提升。这是因为自适应模块是为了对不同的图生成不同的滤波器权重。然而，在顶点为中心的任务中，只有一个图，卷积权重直接就学出来了。因此自适应模块在顶点为中心的任务中是冗余的。
![Table1](/images/convolution-on-graph-a-high-order-and-adaptive-approach/Table1.JPG)

**Graph-centric learning**
预测分子图。目标是给定分子图，预测分子性质。我们使用Duvenaud et al., 2015描述的数据集，评估三种属性：
Solubility, Drug efficacy, Organic photovolatic efficiency。
训练和架构：
l1_gcn和l2_gcn分别表示有一个和两个卷积层的卷积神经网络。我们记录了RMSE（表2）。

Name|Architectures
:-:|:-:
l1_gcn|gcn_{1,2,3}-ReLU-fc64-ReLU-fc16-ReLU-fc1
l1_adp_gcn|adp_gcn{1,2,3}-ReLU-fc64-ReLU-fc16-ReLU-fc1
l2_gcn|[gcb_{1,2,3}-ReLU]*2-fc64-ReLU-fc16-ReLU-fc1
l2_adp_gcn|[adp_gcn_{1,2,3}-ReLU]*2-fc64-ReLU-fc16-ReLU-fc1

![Table2](/images/convolution-on-graph-a-high-order-and-adaptive-approach/Table2.JPG)

node-GCN就是没有自适应滤波模块的一阶HA-GCN。对比node-GCN,l1_gcn,l2_gcn，可以看到我们的卷积层的效果。有自适应滤波器的比没有的效果好。

还有一部分图生成模型，就不说了。