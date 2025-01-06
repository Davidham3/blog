---
categories:
- 论文阅读笔记
date: 2018-04-18 10:43:36+0000
draft: false
math: true
tags:
- deep learning
- machine learning
- computer vision
- graph convolutional network
- Spatial-temporal
- Time Series
title: Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
---
AAAI 2018，以人体关节为图的顶点，构建空间上的图，然后通过时间上的关系，连接连续帧上相同的关节，构成一个三维的时空图。针对每个顶点，对其邻居进行子集划分，每个子集乘以对应的权重向量，得到时空图上的卷积定义。实现时使用Kipf & Welling 2017的方法实现。原文链接：[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455)
<!--more-->
# 摘要
人体骨骼的动态传递了用于人体动作识别的很多信息。传统方法需要手工和遍历规则，导致表现力的限制和泛化的困难。我们提出了动态骨骼识别的新模型，STGCN，可以从数据中自动学习时空模式。这套理论有很强的表达能力与泛化能力。在两个大型数据集Kinetics和NTU-RGBD上比主流方法表现的更好。

# 1 引言
动作识别在视频理解中很有用。一般，从多个角度识别人体动作，如外表、景深、光源、骨骼。对骨骼建模受到的关注较外表和光源较少，我们系统的研究了这个模态，目的是研发出一个有效的对动态骨骼建模的方法，服务于动作识别。

动态骨骼模态很自然地表示成人体关节的时间序列，以2D或3D坐标的形式。人体动作可以通过分析移动规律来识别。早期的工作只用每帧的关节坐标生成特征向量，然后使用空间分析（Wang et al., 2012; Fernando et al., 2015）。这些方法能力受限的原因是他们没有挖掘关节之间的空间信息，但是这些信息对于理解人体动作来说很关键。最近，新的方法尝试利用关节间的自然连接关系(Shahroudy et al., 2016; Du, Wang, and Wang 2015)。这些方法都有提升，表明了连接的重要性。然而，很多显存的方法依赖手工的部分或是分析空间模式的规则。结果导致针对特定问题设计的模型不能泛化。

为了跨越这些限制，我们需要一个新的方法能自动捕获关节的空间配置与时间动态性中嵌入的模式。这就是深度神经网络的优势了。由于骨骼是图结构，不是2D或3D网格，因此传统的CNN不行，最近GCN已经成功的应用在了一些应用上，如图像分类(Bruna et al., 2014)，文档分类(Defferrard, Bresson, and Vandergheynst 2016)，还有半监督学习(Kipf and Welling 2017)。然而，这些工作都假设一个固定的图作为输入。GCN的应用在大尺度的数据集上对动态图建模，如人体骨骼序列还没有被挖掘过。

![Fig1]](/images/spatial-temporal-graph-convolutional-networks-for-skeleton-based-action-recognition/Fig1.JPG)
我们将图网络扩展到一个时空图模型来对骨骼序列进行表示后识别动作。图1所示，这个模型基于一个骨骼图序列，每个顶点表示人体的一个关节。有两种类型的边，空间边，连接关节，时间边连接连续时间的同一关节。构建在上面的时空卷积可以同时集成时间和空间上的信息。

ST-GCN的层次本质消除了手工和遍历部分。不仅有更强的表达能力和更好的表现，也更简单的泛化到其他环境中。在基础的GCN公式基础上，受到图像模型的启发，我们还提出了设计图卷积核新的策略。

我们的工作有三点贡献：
1. 提出了ST-GCN，对动态骨骼建模的基于图结构的模型，第一个应用基于图的神经网络到这个任务上。
2. 设计ST-GCN的卷积核时提出了几个原则使得在骨骼建模时满足特定的需求。
3. 在两个大尺度数据集上，我们提出的模型效果比之前的手工和遍历规则的方法强。
代码和模型：https://github.com/yysijie/st-gcn

# 2 相关工作
两类方法
1. 谱方法，谱分析中考虑图卷积的局部形式(Henaff, Bruna, and LeCun 2015; Duvenaud et al., 2015; Li et al., 2016; Kipf and Welling., 2017)
2. 空间方法，卷积核直接在顶点和他们的邻居上做卷积(Bruna et al., 2014; Niepert, Ahmed, and Kutzkov 2016)。我们的工作follow了第二种方法。我们在空间领域构建CNN滤波器，通过限制滤波器到每个顶点的一阶邻居上。

**骨骼动作识别** 方法可分为手工方法和深度学习方法。第一类是手工设计特征捕获关节移动的动态性。可以是关节轨迹的协方差矩阵(Hussein et al., 2013)，关节的相对位置(Wang et al., 2012)，身体部分的旋转和变换(Vemulapalli, Arrate, and Chellappa 2014)。深度学习的工作使用RNN(Shahroudy et al., 2016; Zhu et al. 2016; Liu et al. 2016; Zhang, Liu and Xiao 2017)和时间CNN(Li et al. 2017; Ke et al. 2017; Kim and Reiter 2017)用端到端的方式学习动作识别模型。这些方法中很多强调了将关节与身体部分建模的重要性。但这些都需要领域知识。我们的ST-GCN是第一个将图卷积用在骨骼动作识别上的。不同于之前的方法，我们的方法可以利用图卷积的局部性和时间动态性学习身体部分的信息。通过消除手工部分标注的需要，模型更容易去设计，而且能学到更好的动作表示。

# 3 Spatial Temporal Graph ConvNet
人们在活动的时候，关节只在一个范围内活动，这个部分称为body parts。已有的方法已经证明了将body parts融入到模型中是很有效的(Shahroudy et al., 2016; Liu et al., 2016; Zhang, Liu and Xiao 2017)。我们认为提升很有可能是因为parts将关节轨迹限制在了局部区域中。像物体识别这样的任务，层次表示和局部性通常是卷积神经网络潜在就可以获得的(Krizhevsky, Sutskever, and Hinton 2012)，而不是手动分配的。这使得我们在基于骨骼的动作识别中引入CNN的性质。结果就是ST-GCN模型的尝试。

## 3.1 Pipeline Overview
骨骼数据通过动作捕捉设备和动作估计算法即可从视频中获得。通常数据是一系列的帧，每帧有一组关节坐标。给定身体关节2D或3D的坐标序列，我们构建了一个时空图，关节作图的顶点，身体结构或时间作边。ST-GCN的输入因此就是关节坐标向量。可以认为这是基于图片的CNN的近似，后者的输入是2D网格中的像素向量。多层时空图卷积操作加到输入上会生成更高等级的特征。然后使用softmax做费雷。整个模型以端到端的方式进行训练。

## 3.2 骨骼图构建
骨骼序列通常表示成每帧都是人体关节的2D或3D坐标。之前使用卷积来做骨骼动作识别的工作(Kim and Reiter 2017)拼接了在每帧拼接了所有关节的坐标向量来生成一个特征向量。我们的工作中，我们利用时空图来生成骨骼序列的层次表示。特别地，我们构建了无向时空图$G = (V, E)$，$N$个关节，$T$帧描述身体内和帧与帧之间的连接。

顶点集$V = \lbrace v\_{ti} \mid t = 1, ..., T, i = 1, ..., N \rbrace$包含了骨骼序列中所有的关节。ST-GCN的输入，每个顶点的特征向量$F(v\_{ti})$由第$t$帧的第$i$个关节的坐标向量组成，还有estimation confidence。构建时空图分为两步，第一步，一帧内的关节通过人体结构连接，如图1所示。然后每个关节在连续的帧之间连接起来。这里不需要人工干预。举个例子，Kinetics数据集，我们使用OpenPose toolbox(Cao et al., 2017b)2D动作估计生成了18个关节，而NTU-RGB+D(Shahroudy et al., 2016)数据集上使用3D关节追踪产生了25个关节。ST-GCN可以在这两种情况下工作，并且提供一致的优越性能。图1就是时空图的例子。
严格来说，边集$E$由两个子集组成，第一个子集描述了每帧骨骼内的连接，表示为$E\_S = \lbrace v\_{ti}v\_{tj} \mid (i, j) \in H \rbrace$，$H$是自然连接的关节的结合。第二个子集是连续帧的相同关节$E\_F = \lbrace  v\_{ti} v\_{(t+1)i} \rbrace$，因此$E\_F$中所有的边对于关节$i$来说表示的是它随时间变化的轨迹。

![Fig2](/images/spatial-temporal-graph-convolutional-networks-for-skeleton-based-action-recognition/Fig2.JPG)

## 3.3 空间图卷积神经网络
时间$\tau$上，$N$个关节顶点$V\_t$，骨骼边集$E\_S(\tau) = \lbrace v\_{ti} v\_{tj} \mid t = \tau, (i, j) \in H \rbrace$。图像上的2D卷积的输入和输出都上2D网格，stride设为1时，加上适当的padding，输出的size就可以不变。给定一个$K \times K$的卷积操作，输入特征$f\_{in}$的channels数是$c$。在空间位置$\mathbf{x}$的单个通道的输出值可以写成：

$$\tag{1}
f\_{out}(\mathbf{x}) = \sum^K\_{h=1} \sum^K\_{w=1} f\_{in}(\mathbf{p}(\mathbf{x}, h, w)) \cdot \mathbf{w}(h, w)
$$

**采样函数**$\mathbf{p} : Z^2 \times Z^2 \rightarrow Z^2$对$\mathbf{x}$的邻居遍历。在图像卷积中，也可表示成$\mathbf{p}(\mathbf{x}, h, w) = \mathbf{x} + \mathbf{p}'(h, w)$。

**权重函数**$\mathbf{w}: Z^2 \rightarrow \mathbb{R}^c$提供了一个$c$维的权重向量，与采出的$c$维输入特征向量做内积。需要注意的是权重函数与输入位置$\mathbf{x}$无关。因此滤波器权重在输入图像上是共享的。图像领域标准的卷积通过对$\mathbf{p}(x)$中的举行进行编码得到。更多解释和应用可以看(Dai et al., 2017)。

图上的卷积是对上式的扩展，输入是空间图$V\_t$。feature map $f^t\_{in}: V\_t \rightarrow R^c$在图上的每个顶点有一个向量。下一步扩展是重新定义采样函数$\mathbf{p}$，权重函数是$\mathbf{w}$。

**采样函数.** 图像中，采样函数$\mathbf{p}(h, w)$定义为中心位置$\mathbf{x}$的邻居像素。图中，我们可以定义相似的采样函数在顶点$v\_{ti}$的邻居集合上$B(v\_{ti}) = \lbrace  v\_{tj} \mid d(v\_{tj}, v\_{ti} \leq D \rbrace)$。这里$d(v\_{tj}, t\_{ti})$表示从$v\_{tj}$到$v\_{ti}$的任意一条路径中最短的。因此采样函数$\mathbf{p}: B(v\_{ti}) \rightarrow V$可以写成：
$$\tag{2}
\mathbf{p}(v\_{ti}, v\_{tj}) = v\_{tj}.
$$
我们令$D = 1$，也就是关节的一阶邻居。更高阶的邻居会在未来的工作中实现。

**权重函数.** 对比采样函数，权重函数在定义时更巧妙。在2D卷积，网格型自然就围在了中心位置周围。所以像素与其邻居有个固定的顺序。权重函数根据空间顺序通过对维度为$(c, K, K)$的tensor添加索引来实现。对于像我们构造的这种图，没有这种暗含的关系。解决方法由(Niepert, Ahmed, and Kuzkov 2016)提出，顺序是通过根节点周围的邻居节点的标记顺序确定。我们根据这个思路构建我们的权重函数。不再给每个顶点一个标签，我们通过将顶点$v\_{ti}$的邻居集合$B(v\_{ti})$划分为$K$个子集来简化过程，其中每个子集都有一个数值型标签。因此我们可以得到一个映射$l\_{ti}:B(v\_{ti}) \rightarrow \lbrace 0,...,K-1 \rbrace$，这个映射将顶点映射到它的邻居子集的标签上。权重函数$\mathbf{w}(v\_{ti}, v\_{tj}):B(v\_{ti}) \rightarrow R^c$可以通过对维度为$(c, K)$的tensor标记索引或
$$\tag{3}
\mathbf{w}(v\_{ti}, v\_{tj}) = \mathbf{w}'(l\_{ti}(v\_{tj})).
$$
我们会在3.4节讨论分区策略。

**空间图卷积.** 我们可以将式1重写为：
$$\tag{4}
f\_{out}(v\_{ti}) = \sum\_{v\_{tj} \in B(v\_{ti})} \frac{1}{Z\_{ti}(v\_{tj})} f\_{in}(\mathbf{p}(v\_{ti}, v\_{tj})) \cdot \mathbf{w}(v\_{ti}, v\_{tj}),
$$
其中归一化项$Z\_{ti}(v\_{tj}) = \vert \lbrace  v\_{tk} \mid l\_{ti}(v\_{tk}) = l\_{ti}(t\_{tj}) \rbrace \vert$等于对应子集的基数。这项被加入是来平衡不同子集对输出的贡献。
替换式2和式3，我们可以得到
$$\tag{5}
f\_{out}(v\_{ti}) = \sum\_{v\_{tj} \in B(v\_{ti})} \frac{1}{Z\_{ti}(v\_{tj})} f\_{in}(v\_{tj}) \cdot \mathbf{w}(l\_{ti}(v\_{tj})).
$$
这个公式与标准2D卷积相似如果我们将图片看作2D网格。比如，$3 \times 3$卷积核的中心像素周围有9个像素。邻居集合应被分为9个子集，每个子集有一个像素。

**时空建模.** 通过对空间图CNN的构建，我们现在可以对骨骼序列的时空动态性进行建模。回想图的构建，图的时间方面是通过在连续帧上连接相同的关节进行构建的。这可以让我们定义一个很简单的策略来扩展空间图CNN到时空领域。我们扩展邻居的概念到包含空间连接的关节：
$$\tag{6}
B(v\_{ti}) = \lbrace  v\_{qj} \mid d(v\_{tj}, v\_{ti}) \leq K, \vert q - t \vert \leq \lfloor \Gamma / 2 \rfloor \rbrace.
$$
参数$\Gamma$控制被包含到邻居图的时间范围，因此被称为空间核的大小。我们需要采样函数来完成时空图上的卷积操作，与只有空间卷积一样，我们还需要权重函数，具体来说就是映射$l\_{ST}$。因为空间轴是有序的，我们直接修改根节点为$v\_{ti}$的时空邻居的标签映射$l\_{ST}$为：
$$\tag{7}
l\_{ST}(v\_{qj}) = l\_{ti}(v\_{tj}) + (q - t + \lfloor \Gamma / 2 \rfloor) \times K,
$$
其中$l\_{ti}(v\_{tj})$是$v\_{ti}$的单帧的标签映射。这样，我们就有了一个定义在时空图上的卷积操作。

## 3.4 分区策略
设计一个实现标记映射的分区策略很重要。我们探索了几种分区策略。简单来说，我们只讨论单帧情况下，因为使用式7就可以很自然的扩展到时空领域。

**Uni-labeling.** 最简单的分区策略，所有的邻居都是一个集合。每个邻居顶点的特征向量会和同一个权重向量做内积。事实上，这个策略和Kipf and Welling 2017提出的传播规则很像。但是有个很明显的缺陷，在单帧的时候使用这种分区策略就是将邻居的特征向量取平均后和权重向量做内积。在骨骼序列分析中不能达到最优，因为丢失了局部性质。$K = 1$，$l\_{ti}(v\_{tj}) = 0, \forall{i, j} \in V$。

**Distance partitioning.** 另一个自然的分区策略是根据顶点到根节点$v\_{ti}$的距离$d(\cdot, v\_{ti})$来划分。我们设置$D = 1$，邻居集合会被分成两个子集，$d = 0$表示根节点子集，其他的顶点是在$d = 1$的子集中。因此我们有两个不同的权重向量，他们能对局部性质进行建模，比如关节间的相对变换。$K = 2$，$l\_{ti}(v\_{tj}) = d(v\_{tj}, v\_{ti})$

**Spatial configuration partitioning.** 因为骨骼是空间局部化的，我们仍然可以利用这个特殊的空间配置来分区。我们将邻居集合分为三部分：1. 根节点自己；2. 中心组：相比根节点更接近骨骼重心的邻居顶点；3. 其他的顶点。其中，中心定义为一帧中骨骼所有的关节的坐标的平均值。这是受到人体的运动大体分为同心运动和偏心运动两类。
$$\tag{8}
l\_{ti}(v\_tj) = \begin{cases}
0 & if r\_j = r\_i \\
1 & if r\_j < r\_i \\
2 & if r\_j > r\_i \end{cases}
$$
其中，$r\_i$是训练集中所有帧的重心到关节$i$的平均距离。
分区策略如图3所示。我们通过实验检验提出的分区策略在骨骼动作识别上的表现。分区策略越高级，效果应该是越好的。
![Fig3](/images/spatial-temporal-graph-convolutional-networks-for-skeleton-based-action-recognition/Fig3.JPG)

## 3.5 Learnable edge importance weighting
尽管人在做动作时关节是以组的形式移动的，但一个关节可以出现在身体的多个部分。然而，这些表现在建模时应该有不同的重要性。我们在每个时空图卷积层上添加了一个可学习的mask$M$。这个mask会基于$E\_S$中每个空间图边上可学习的重要性权重来调整一个顶点的特征对它的邻居顶点的贡献。通过实验我们发现增加这个mask可以提升ST-GCN的性能。使用注意力映射应该也是可行的，这个留到以后再做。

## 3.6 Implementation ST-GCN
实现这个图卷积不像实现2D或3D卷积那样简单。我们提供了实现ST-GCN的具体细节。
我们采用了Kipf & Welling 2017的相似的实现方式。单帧内身体内关节的连接表示为一个邻接矩阵$\rm{A}$，单位阵$\rm{I}$表示自连接。在单帧情况下，ST-GCN使用第一种分区策略时可以实现为：
$$\tag{9}
\rm 
f\_{out} = \Lambda^{-\frac{1}{2}}(A + I) \Lambda^{-\frac{1}{2}} f\_{in}W,
$$
其中，$\Lambda^{ii} = \sum\_j (A^{ij} + I^{ij})$。多个输出的权重向量叠在一起形成了权重矩阵$\mathrm{W}$。实际上，在时空情况下，我们可以将输入的feature map表示为维度为$(C, V, T)$的tensor。图卷积通过一个$1 \times \Gamma$实现一个标准的2D卷积，将结果与归一化的邻接矩阵$\rm \Lambda^{-\frac{1}{2}}(A + I)\Lambda^{-\frac{1}{2}}$在第二个维度上相乘。

对于多个子集的分区策略，我们可以再次利用这种实现。但注意现在邻接矩阵已经分解成了几个矩阵$A\_j$，其中$\rm A + I = \sum\_j A\_j$。举个例子，在距离分区策略中，$\rm A\_0 = I$，$\rm A\_1 = A$。式9变形为
$$\tag{10}
\rm f\_{out} = \sum\_j \Lambda^{-\frac{1}{2}}\_j A\_j \Lambda^{\frac{1}{2}}\_j f\_{in} W\_j
$$
其中，$\rm \Lambda^{ii}\_j = \sum\_k (A^{ik}\_j) + \alpha$。这里我们设$\alpha = 0.001$避免$\rm A\_j$中有空行。

实现可学习的边重要性权重很简单。对于每个邻接矩阵，我们添加一个可学习的权重矩阵$M$，替换式9中的$\rm A + I$和式10中的$\rm A\_j$中的$A\_j$为$\rm (A + I) \otimes M$和$\rm A\_j \otimes M$。这里$\otimes$表示两个矩阵间的element-wise product。mask$M$初始化为一个全一的矩阵。