---
categories:
- 论文阅读笔记
date: 2018-05-10 15:35:47+0000
description: 'IJCAI 2018，大体思路：使用Kipf & Welling 2017的近似谱图卷积得到的图卷积作为空间上的卷积操作，时间上使用一维卷积对所有顶点进行卷积，两者交替进行，组成了时空卷积块，在加州PeMS和北京市的两个数据集上做了验证。但是图的构建方法并不是基于实际路网，而是通过数学方法构建了一个基于距离关系的网络。原文链接：[Spatio-Temporal
  Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/abs/1709.04875v4)'
draft: false
math: true
tags:
- deep learning
- ResNet
- Spatial-temporal
- graph convolutional network
- Graph
- Time Series
- 已复现
title: 'Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for
  Traffic'
---
IJCAI 2018，大体思路：使用Kipf & Welling 2017的近似谱图卷积得到的图卷积作为空间上的卷积操作，时间上使用一维卷积对所有顶点进行卷积，两者交替进行，组成了时空卷积块，在加州PeMS和北京市的两个数据集上做了验证。但是图的构建方法并不是基于实际路网，而是通过数学方法构建了一个基于距离关系的网络。原文链接：[Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/abs/1709.04875v4)
<!--more-->
# 摘要
实时精确的交通预测对城市交通管控和引导很重要。由于交通流的强非线性以及复杂性，传统方法并不能满足中长期预测的要求，而且传统方法经常忽略对时空数据的依赖。在这篇文章中，我们提出了一个新的深度学习框架，时空图卷积(Spatio-Temporal Graph Convolutional Networks)，来解决交通领域的时间序列预测问题。我们在图上将问题形式化，并且建立了完全卷积的结构，并不是直接应用传统的卷积以及循环神经单元，这可以让训练速度更快，参数更少。实验结果显示通过在多尺度的交通网络上建模，STGCN模型可以有效地捕获到很全面的时空相关性并且在各种真实数据集上表现的要比很多state-of-the-art算法好。

# 引言
交通运输在每个人的生活中都扮演着重要的角色。根据2015年的调查，美国的司机们平均每天要在车上呆48分钟。这种情况下，精确的实时交通状况预测对于路上的用户，private sector和政府来说变得至关重要。广泛使用的交通服务，如交通流控制、路线规划和导航，也依赖于高质量的交通状况预测。总的来说，多尺度的交通预测的研究很有前景而且是城市交通流控制和引导的基础，也是智能交通系统的一个主要功能。

在交通研究中，交通流的基本变量，也就是速度、流量和密度，通常作为监控当前交通状态以及未来预测的指示指标。根据预测的长度，交通预测大体分为两个尺度：短期(5~30min)，中和长期预测(超过30min)。大多数流行的统计方法(比如，线性回归)可以在短期预测上表现的很好。然而，由于交通流的不确定性和复杂性，这些方法在相对长期的预测上不是那么的有效。

之前在中长期交通预测上的研究可以大体的分为两类：动态建模和数据驱动的方法。动态建模使用了数学工具（比如微分方程）和物理知识通过计算模拟来形式化交通问题[Vlahogiani, 2015]。为了达到一个稳定的状态，模拟进程不仅需要复杂的系统编程，还需要消耗大量的计算资源。模型中不切实际的假设和化简也会降低预测的精度。因此，随着交通数据收集和存储技术的快速发展，一大群研究者正在将他们的目光投向数据驱动的方法。
典型的统计学和机器学习模型是数据驱动方法的两种体现。在时间序列分析上，自回归移动平均模型（ARIMA）和它的变形是众多统一的方法中基于传统统计学的方法[Ahmed and Cook, 1979; Williams and Hoel, 2003; Lippi $et al.$, 2013]。然而，这种类型的模型受限于时间序列的平稳分布，而且不能考虑时空相关性。因此，这些方法限制了高度非线性的交通流的表示能力。最近，传统的统计方法在交通预测上已经受到了机器学习方法的冲击。这些模型可以获得更高的精度，对更复杂的数据建模，比如k近邻（KNN），支持向量机（SVM）和神经网络（NN）。

**深度学习方法** 最近，深度学习已经被广泛且成功地应用于各式各样的交通任务中，在最近的工作中已经取得了很显著的成果，比如，深度置信网络（DBN）[Jia *et al.*, 2016; Huang *et al.*, 2014]和层叠自编码器(stacked autoencoder)(SAE)[Lv *et al.*, 2015; Chen *et al.*, 2016]。然而，这些全连接神经网络很难从输入中提取空间和时间特征。而且，空间属性的严格限制甚至完全缺失，这些网络的表示能力被限制的很严重。

为了充分利用空间特征，一些研究者使用了卷积神经网络来捕获交通网络中的临近信息，同时也在时间轴上部署了循环神经网络。通过组合长短期记忆网络[Hochreiter and Schmidhuber, 1997]和1维卷积，Wu和Tan[2016]首先提出一个特征层面融合的架构CLTFP来预测短期交通状况。尽管它采取了一个很简单的策略，CLTFP仍然是第一个尝试对时间和空间规律性对齐的方法。后来，Shi *et al.*[2015]提出了卷积LSTM，这是一个带有嵌入卷积层的全连接LSTM的扩展。然而，常规的卷积操作限制了模型只能处理常规的网格结构（如图像或视频），而不是其他的大部分领域（比如Graph）。与此同时，循环神经网络对于序列的学习需要迭代训练，这会导致误差的积累。更进一步地说，循环神经网络（包括基于LSTM的RNN）的难以训练和计算量大是众所周知的。

为了克服这些问题，我们引入了一些策略来有效的对交通流的时间动态和空间依赖进行建模。为了完全利用空间信息，我们通过一个广义图对交通网络建模，而不是将交通流看成各个离散的部分（比如网格或碎块）。为了处理循环神经网络的缺陷，我们在时间轴上部署了一个全卷积结构来阻止累积效应（cumulative effects）并且加速模型的训练过程。综上所述，我们提出了一个新的神经网络架构，时空图卷积网络，来预测交通情况。这个架构由多个时空图卷积块组成，这些都是图卷积层和卷积序列学习层（convolutional sequence learning layers）的组合，用来对时间和空间依赖关系进行建模。

我们的主要贡献可以归纳为以下三点：
1. 我们研究了在交通领域时间与空间依赖结合的好处。为了充分利用我们的知识，这是在交通研究中第一次应用纯卷积层来同时从图结构的时间序列中提取时空信息。
2. 我们提出了一个新的由时空块组成的神经网络结构。由于这个架构中是纯卷积操作，它比基于RNN的模型的训练速度快10倍以上，而且需要的参数更少。这个架构可以让我们更有效地处理更大的路网，这部分将在第四部分展示。
3. 我们在两个真实交通数据集上验证了提出来的网络。这个实验显示出我们的框架比已经存在的在多长度预测和网络尺度上的模型表现的更好。

# 准备工作
## 路网上的交通预测
交通预测是一个典型的时间序列预测问题，也就是预测在给定前M个观测样本接下来H个时间戳后最可能的交通流指标（比如速度或交通流），

$$

	\tag{1}	\hat{v}\_{t+1}, ..., \hat{v}\_{t+H} = \mathop{\arg\min}\_{v\_{t+1},...,v\_{t+H}}logP(v\_{t+1},...,v\_{t+H}\vert v\_{t-M+1},...v\_t)
$$

这里$v\_t \in \mathbb{R}^n$是$n$个路段在时间戳$t$观察到的一个向量，每个元素记录了一条路段的历史观测数据。

在我们的工作中，我们在一个图上定义了一个交通网络，并专注于结构化的交通时间序列。观测到的样本$v\_t$间不是相互独立的，而是在图中两两相互连接的。因此，数据点$v\_t$可以被视为定义在权重为$w\_{ij}$，如图1展示的无向图（或有向图）$\mathcal{G}$上的一个信号。在第$t$个时间戳，在图$\mathcal{G\_t}=(\mathcal{V\_t}, \mathcal{\varepsilon}, W)$, $\mathcal{V\_t}$是当顶点的有限集，对应在交通网络中$n$个监测站；$\epsilon$是边集，表示观测站之间的连通性；$W \in \mathbb{R^{n \times n}}$表示$\mathcal{G\_t}$的邻接矩阵。

![Fig1](/blog/images/spatio-temporal-graph-convolutional-networks-a-deep-learning-framework-for-traffic/Fig1.PNG)

## 图上的卷积

传统网格上的标准卷积很明显是不能应用在广义图上的。现在有两个基本的方法正在探索如何泛化结构化数据上的CNN。一个是扩展卷积的空间定义[Niepert *et al.*, 2016]，另一个是使用图傅里叶变换在谱域中进行操作[Bruna *et al.*, 2013]。前一个方法重新将顶点安排至确定的表格形式内，然后就可以使用传统的卷积方法了。后者引入了谱框架，在谱域中应用图卷积，经常被称为谱图卷积。一些后续的研究通过将时间复杂度从$O(n^2)$降至线性[Defferrard *et al.*, 2016;Kipf and Welling, 2016]使谱图卷积的效果更好。
我们基于谱图卷积的定义引入图卷积操作“$\ast\_{\mathcal{G}}$”的符号，也就是一个核$\Theta$和信号$x \in \mathbb{R}^n$的乘法，

$$\tag{2} \Theta \ast\_{\mathcal{G}}x=\Theta(L)x=\Theta(U \Lambda U^T)x=U\Theta(\Lambda)U^Tx$$

这里图的傅里叶基$U \in \mathbb{R}^{n \times n}$是归一化的拉普拉斯矩阵$L=I\_n-D^{-1/2}WD^{-1/2}= U \Lambda U^T \in \mathbb{R}^{n \times n}$的特征向量组成的矩阵，其中$I\_n$是单位阵，$D \in \mathbb{R}^{n \times n}$是对角的度矩阵$D\_{ii}=\sum\_j{W\_{ij}}$；$\Lambda \in \mathbb{R}^{n \times n}$是$L$的特征值组成的矩阵，卷积核$\Theta(\Lambda)$是一个对角矩阵。通过这个定义，一个图信号$x$是被一个核$\Theta$通过$\Theta$和图傅里叶变换$U^Tx$[Shuman *et al.*, 2013]过滤的。

# 提出的模型

## 网络架构

在这部分，我们详细说明了时空图卷积网络的框架。如图二所示，STGCN有多个时空卷积块组成，每一个都是像一个“三明治”结构的组成，有两个门序列卷积层和一个空间图卷积层在中间。每个模块的细节如下。

![Fig2](/blog/images/spatio-temporal-graph-convolutional-networks-a-deep-learning-framework-for-traffic/Fig2.PNG)

图二：时空图卷积网络的架构图。STGCN的架构有两个时空卷积块和一个全连接的在末尾的输出层组成。每个ST-Conv块包含了两个时间门卷积层，中间有一个空间图卷积层。每个块中都使用了残差连接和bottleneck策略。输入$v\_{t-M+1},...v\_t$被ST-Conv块均匀的（uniformly）处理，来获取时空依赖关系。全部特征由一个输出层来整合，生成最后的预测$\hat{v}$。

## 提取空间特征的图卷积神经网络

交通网络大体上是一个图结构。由数学上的图来构成路网是很自然也很合理的。然而，之前的研究忽视了交通网络的空间属性：因为交通网络被分成了块或网格状，所以网络的全局性和连通性被过分的关注了。即使是在网格上的二维卷积，由于数据建模的折中，也只能捕捉到大体的空间局部性。根据以上情况，在我们的模型中，图卷积被直接的应用在了图结构数据上为了在空间中抽取很有意义的模式和特征。集是在图卷积中由式2可以看出核$\Theta$的计算的时间复杂度由于傅里叶基的乘法可以达到$O(n^2)$，两个近似的策略可以解决这个问题。

**切比雪夫多项式趋近**

为了局部化过滤器并且减少参数，核$\Theta$可以被一个关于$\Lambda$的多项式限制起来，也就是$\Theta(\Lambda)=\sum\_{k=0}^{K-1} \theta\_k \Lambda^k$，其中$\theta \in \mathbb{R}^K$是一个多项式系数的向量。$K$是图卷积核的大小，它决定了卷积从中心节点开始的最大半径。一般来说，切比雪夫多项式$T\_k(x)$被用于近似核，作为$K-1$阶展开的一部分，也就是$\Theta(\Lambda) \approx \sum\_{k=0}^{K-1} \theta\_k T\_k(\widetilde{\Lambda})$，其中$\widetilde{\Lambda}=2\Lambda/\lambda\_{max}-I\_n$（$\lambda\_{max}$表示$L$的最大特征值）[Hammond *et al.*, 2011]。图卷积因此可以被写成

$$
\tag{3} \Theta \ast\_{\mathcal{G}} x = \Theta(L)x \approx \sum\_{k=0}^{K-1}\theta\_k T\_k(\widetilde{L})x
$$

其中$T\_k(\widetilde{L}) \in \mathbb{R}^{n \times n}$是k阶切比雪夫多项式对缩放后（scaled）的拉普拉斯矩阵$\widetilde{L}=2L/\lambda\_{max}-I\_n$。通过递归地使用趋近后的切比雪夫多项式计算K阶卷积操作，式2的复杂度可以被降低至$O(K\vert \varepsilon \vert)$，如式3所示[Defferrard *et al.*, 2016]。

**1阶近似**

一个针对层的线性公式可以由堆叠多个使用拉普拉斯矩阵的一阶近似的局部图卷积层[Kipf and Welling, 2016]。结果就是，这样可以构建出一个深的网络，这个网络可以深入地恢复空间信息并且不需要指定多项式中的参数。由于在神经网络中要缩放和归一化，我们可以进一步假设$\lambda\_{max} \approx 2$。因此，式3可以简写为

$$
\begin{aligned}
\Theta \ast\_{\mathcal{G}}x \approx & \theta\_0x+\theta\_1(\frac{2}{\lambda\_{max}}L-I\_n)x\\ 
						 \approx & \theta\_0 x- \theta\_1(D^{-\frac{1}{2}} W D^{-\frac{1}{2}}) x
\end{aligned}
$$

其中，$\theta\_0$，$\theta\_1$是核的两个共享参数。为了约束参数并为稳定数值计算，$\theta\_0$和$\theta\_1$用一个参数$\theta$来替换，$\theta=\theta\_0=-\theta\_1$；$W$和$D$是通过$\widetilde{W}=W+I\_n$和$\widetilde{D}\_{ii}=\sum\_j\widetilde{W}\_{ij}$重新归一化得到的。之后，图卷积就可以表达为

$$
\begin{aligned}
\Theta \ast\_{\mathcal{G}} x = & \theta(I\_n + D^{-\frac{1}{2}} W D^{\frac{1}{2}})x\\
= & \theta (\widetilde{D}^{-\frac{1}{2}} \widetilde{W} \widetilde{D}^{-\frac{1}{2}})x
\end{aligned}
$$

竖直地堆叠一阶近似的图卷积可以获得和平行的K阶卷积相同的效果，所有的卷积可以从一个顶点的$K-1$阶邻居中获取到信息。在这里，$K$是连续卷积操作的次数或是模型中的卷积层数。进一步说，针对层的线性结构是节省参数的，并且对大型的图来说是效率很高的，因为多项式趋近的阶数为1。

**图卷积的泛化**

图卷积操作$\ast\_{\mathcal{G}}$也可以被扩展到多维张量上。对于一个有着$C\_i$个通道的信号$X \in \mathbb{R}^{n \times C\_i}$，图卷积操作可以扩展为

$$
y\_j = \sum\_{i=1}^{C\_i} \Theta\_{i,j}(L) x\_i \in \mathbb{R}^n, 1 \leq j \leq C\_o
$$

其中，$C\_i \times C\_o$个向量是切比雪夫系数$\Theta\_{i,j} \in \mathbb{R}^K$（$C\_i$，$C\_o$分别是feature map的输入和输出大小）。针对二维变量的图卷积表示为$\Theta \ast\_{\mathcal{G}} X$，其中$\Theta \in \mathbb{R}^{K \times C\_i \times C\_o}$。需要注意的是，输入的交通预测是由$M$帧路网组成的，如图1所示。每帧$v\_t$可以被视为一个矩阵，它的第$i$列是图$\mathcal{G\_t}$中第$i$个顶点的一个为$C\_i$维的值，也就是$X \in \mathbb{R}^{n \times C\_i}$（在这个例子中，$C\_i=1$）。对于$M$中的每个时间步$t$，相同的核与相同的图卷积操作是在$X\_t \in \mathbb{R}^{n \times C\_i}$中并行进行的。因此，图卷积操作也可以泛化至三维，记为$\Theta \ast\_{\mathcal{G}} \mathcal{X}$，其中$\mathcal{X} \in \mathbb{R}^{M \times n \times C\_i}$

## 抽取时间特征的门控卷积神经网络

尽管基于RNN的模型可以广泛的应用于时间序列分析，用于交通预测的循环神经网络仍然会遇到费时的迭代，复杂的门控机制，对动态变化的响应慢。相反，CNN训练快，结构简单，而且不依赖于前一步。受到[Gehring *et al.*, 2017]的启发，我们在时间轴上部署了整块的卷积结构，用来捕获交通流的动态时间特征。这个特殊的设计可以让并行而且可控的训练过程通过多层卷积结构形成层次表示。

如图2右侧所示，时间卷积层包含了一个一维卷积，核的宽度为$K\_t$，之后接了一个门控线性单元(GLU)作为激活。对于图$\mathcal{G}$中的每个顶点，时间卷积对输入元素的$K\_t$个邻居进行操作，导致每次将序列长度缩短$K\_t-1$。因此，每个顶点的时间卷积的输入可以被看做是一个长度为$M$的序列，有着$C\_i$个通道，记作$Y \in \mathbb{R}^{M \times C\_i}$。卷积核$\Gamma \in \mathbf{R}^{K\_t \times C\_i \times 2C\_o}$是被设计为映射$Y$到一个单个的输出$[P Q] \in \mathbb{R}^{(M-K\_t+1) \times (2C\_o)}$($P$, $Q$是通道数的一半)。作为结果，时间门控卷积可以定义为：

$$
\Gamma \ast\_ \tau Y = P \otimes \sigma (Q) \in \mathbb{R}^{(M-K\_t+1) \times C\_o}
$$

其中，$P$, $Q$分别是GLU的输入门，$\otimes$表示哈达玛积，sigmoid门$\sigma(Q)$控制当前状态的哪个输入$P$对于发现时间序列中的组成结构和动态方差是相关的。非线性门通过堆叠时间层对挖掘输入也有贡献。除此以外，在堆叠时间卷积层时，实现了残差连接。相似地，通过在每个节点$\mathcal{Y\_i} \in \mathbb{R}^{M \times C\_i}$(比如监测站)上都使用同样的卷积核$\Gamma$，时间卷积就可以泛化至3D变量上，记作$\Gamma \ast\_\tau \mathcal{Y}$，其中$\mathcal{Y} \in \mathbb{R}^{M \times n \times C\_i}$。  

**这里我之前认为残差是用了 padding 的，其实不是，看了作者的代码后发现作者是用了一半数量的卷积核完成卷积，这样就和 P 的维度一致了，然后直接和 P 相加，然后与 sigmoid 激活后的值进行点对点的相乘。**

## 时空卷积块

为了同时从空间和时间领域融合特征，时空卷积块(ST-Conv block)的构建是为了同时处理图结构的时间序列的。如图2（中）所示，bottleneck策略的应用形成了三明治的结构，其中含有两个时间门控卷积层，分别在上下两层，一个空间图卷积层填充中间的部分。空间卷积层导致的通道数$C$的减小促使了参数的减少，并且减少了训练的时间开销。除此以外，每个时空块都使用了层归一化来抑制过拟合。

ST-Conv块的输入和输出都是3D张量。对于块$l$的输入$v^l \in \mathbb{R}^{M \times n \times C^l}$，输出$v^{l+1} \in \mathbb{R}^{(M-2(K\_t-1)) \times n \times C^{l+1}}$通过以下式子计算得到：

$$
v^{l+1} = \Gamma^l\_1 \ast\_\tau \rm ReLU(\Theta^l \ast\_{\mathcal{G}}(\Gamma^l\_0 \ast\_\tau v^l))
$$

其中$\Gamma^l\_0$，$\Gamma^l\_1$是块$l$的上下两个时间层；$\Theta^l$是图卷积谱核；$\rm ReLU(·)$表示ReLU激活函数。我们在堆叠两个ST-Conv块后，加了一个额外的时间卷积和全连接层作为最后的输出层（图2左侧）。时间卷积层将最后一个ST-Conv块的输出映射到一个最终的单步预测上。之后，我们可以从模型获得一个最后的输出$Z \in \mathbb{R}^{n \times c}$，通过一个跨$c$个通道的线性变换$\hat{v} = Zw+b$来预测$n$个节点的速度，其中$w \in \mathbb{R}^c$是权重向量,$b$是偏置。对交通预测的STGCN的损失函数可以写成：

$$
L(\hat{v}; W\_\theta) = \sum\_t \Vert \hat{v}(v\_{t-M+1, ..., v\_t, W\_\theta}) - v\_{t+1} \Vert^2
$$

其中，$W\_\theta$是模型中所有的训练参数; $v\_{t+1}$是ground truth，$\hat{v}(·)$表示模型的预测。

我们来总结一下我们的STGCN的主要特征：

1. STGCN是处理结构化的时间序列的通用框架，不仅可以解决交通网络建模，还可以应用到其他的时空序列学习的挑战中，比如社交网络和推荐系统。
2. 时空块融合了图卷积和门控时间卷积，可以同时抽取有用的空间信息，捕获本质上的时间特征。
3. 模型完全由卷积层组成，因此可以在输入序列上并行运算，空间域中参数少易于训练。更重要的是，这个经济的架构可以使模型更高效的处理大规模的网络。

# 实验

## 数据集描述

我们在两个真实的数据集上验证了模型，分别是**BJER4**和**PeMSD7**，由北京市交委和加利福尼亚运输部提供。每个数据集包含了交通观测数据的关键属性和对应时间的地图信息。

**BJER4**是通过double-loop detector获取的东四环周边的数据。我们的实验中有12条道路。交通数据每五分钟聚合一次。时间是从2014年的7月1日到8月31日，不含周末。我们选取了第一个月的车速速度记录作为训练集，剩下的分别做验证和测试。

**PeMSD7**是Caltrans Performance Measurement System(PeMS)通过超过39000个监测站实时获取的数据，这些监测站分布在加州高速公路系统主要的都市部分[Chen *et al*., 2001]。数据是30秒的数据样本聚合成5分钟一次的数据。我们在加州的District 7随机选取了一个小的和一个大的范围作为数据源，分别有228和1026个监测站，分别命名为PeMSD7(S)和PeMSD7(L)（如图3左侧所示）。PeMSD7数据集的时间范围是2012年五月和六月的周末。我们使用同样的原则对数据进行了训练集和测试集的划分。

![Fig1](/blog/images/spatio-temporal-graph-convolutional-networks-a-deep-learning-framework-for-traffic/Fig3.PNG)

## 数据预处理

两个数据集的间隔设定为5分钟。因此，路网中的每个顶点每天就有288个数据点。数据清理后使用了线性插值的方法来填补缺失值。通过核对相关性，每条路的方向和OD(origin-destination)点，环路系统可以被数值化成一个有向图。

在PeMSD7，路网的邻接矩阵通过交通网络中的监测站的距离来计算。带权邻接矩阵$W$通过以下公式计算：

$$
w\_{ij} = \begin{cases}
\exp{(-\frac{d^2\_{ij}}{\sigma^2})}&,i \neq j \ \rm and \exp{(-\frac{d^2\_{ij}}{\sigma^2}) \geq \epsilon} \\
0&, \rm otherwise
\end{cases}
$$

其中$w\_{ij}$是边的权重，通过$d\_{ij}$得到，也就是$i$和$j$之间的距离。$\sigma^2$和$\epsilon$是来控制矩阵$W$的分布和稀疏性的阈值，我们用了10和0.5。$W$的可视化在图3的右侧。

# 代码
[作者代码](https://github.com/VeritasYin/STGCN_IJCAI-18)，这个是作者提供的代码。

[仓库地址](https://github.com/Davidham3/STGCN)，我按照论文结合作者的代码进行了复现与修正。