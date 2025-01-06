---
categories:
- 论文阅读笔记
date: 2018-12-18 20:49:15+0000
draft: false
math: true
tags:
- deep learning
- Graph
- graph convolutional network
title: Geometric deep learning on graphs and manifolds using mixture model CNNs
---
CVPR 2017. 这篇论文有点难，没看下去。。。原文链接：[Geometric deep learning on graphs and manifolds using mixture model CNNs](https://arxiv.org/abs/1611.08402.pdf)
<!--more-->

# Abstract 

大部分深度学习处理的是 1D，2D，3D 欧式结构数据，音频信号、图像、视频。最近大家开始研究在非欧氏空间上的数据，如复杂网络、计算社会科学、计算机图形学。我们提出了一个统一的框架，让 CNN 可以泛化到非欧氏空间上，学习局部的、平稳的、针对任务可分解的特征。我们发现之前提出的一些方法都可以放到我们的框架中。我们发现我们的方法效果比前人的方法都要好。

# 1. Introduction

# 2. Deep learning on graphs

无向带权图 $\mathcal{G} = (\lbrace 1, \dots, n\rbrace, \mathcal{E}, \mathbf{W})$，邻接矩阵 $\mathbf{W} = (w\_{ij})$，其中 $w\_{ij} = w\_{ji}$，如果 $(i, j) \notin \mathcal{E}$，则 $w\_{ij} = 0$，否则 $w\_{ij} > 0$。未归一化的拉普拉斯矩阵是个 $n \times n$ 的 实对称半正定矩阵 $\Delta = \bf D - W$，其中 $\mathbf{D} = \text{diag}(\sum\_{j = \not i} w\_{ij})$ 是度矩阵。

拉普拉斯矩阵有特征值分解 $\bf \Delta = \Phi \Lambda \Phi^T$，其中 $\Phi = (\phi\_1, \dots, \phi\_n)$ 是相互正交的特征向量，$\Lambda = \text{diag}(\lambda\_1, ..., \lambda\_n)$ 特征值组成的对角矩阵。在传统的谐波分析中，特征向量是拉普拉斯算子，特征值可以看作是频率。给定图上的一个信号 $\mathbf{f} = (f\_1, \dots, f\_n)^T$，它的图傅里叶变换是 $\hat{\mathbf{f}} = \Phi^T \mathbf{f}$。给定两个信号 $\bf f, g$，他们的谱卷积定义为傅里叶变换的 element-wise product：

$$\tag{1}
\mathbf{f} \star \mathbf{g} = \Phi (\Phi^T \mathbf{f}) \odot (\Phi^T g) = \Phi \ \text{diag}(\hat{g}\_1, \dots, \hat{g}\_n) \hat{f},
$$

对应了欧氏空间卷积理论。

**其实这里我没理解啊，我记得卷积的定义不是傅里叶变换的乘积的逆变换吗，所以感觉说的有点不对，但公式倒是对了。。。**

**Spectral CNN.** Bruna et al. 使用卷积在谱上的定义将 CNN 泛化到图上，得到一个谱卷积层的定义：

$$\tag{2}
\mathbf{f}^{out}\_l = \xi (\sum^p\_{l'=1} \Phi\_k \hat{G}\_{l,l'} \Phi^T\_k \mathbf{f}^{in}\_{l'})
$$

这里维数为 $n \times p$ 和 $n \times q$ 的矩阵 $\mathbf{F}^{in} = (\mathbf{f}^{in}\_1, \dots, \mathbf{f}^{in}\_p)$，$\mathbf{F}^{out} = (\mathbf{f}^{out}\_1, \dots, \mathbf{f}^{out}\_q)$ 分别表示 $p$ 维和 $q$ 维的图上的输入和输出信号，$\Phi = (\phi\_1, \dots, \phi\_k)$ 是前几个特征向量组成的 $n \times k$ 的矩阵，$\hat{\mathbf{G}\_{l,l'}} = \text{diag}(\hat{g}\_{l,l',1}, \dots, \hat{g}\_{l,l',k})$ 是一个 $k \times k$ 的对角矩阵，表示频域内一个可学习的滤波器，$\xi$ 是一个非线性激活单元（e.g. ReLU）。这个框架的池化操作在图上的模拟是一个图的缩减操作，给定一个 $n$ 个结点的图，生成一个 $n' < n$ 个结点的图，将信号从原来的图上变换到缩减后的图上。

这个框架有几个缺点。首先，谱滤波器的系数是 *basis dependent*，而且，在一个图上学习到的基于谱的 CNN 模型不能应用在其他的图上。其次，图傅里叶变换的计算因为 $\bf \Phi$ 和 $\bf \Phi^T$ 的乘法，会达到 $\mathcal{O}(n^2)$，因为这里没有像 FFT 一样的算法。第三，不能保证在谱域内的滤波器在顶点域上是局部化的；假设使用 $k = O(n)$ 个归一化的拉普拉斯矩阵的特征向量，一个谱卷积层需要 $pqk = O(n)$ 个参数。

**Smooth Spectral CNN.** 之后，Henaff et al. 认为 smooth 谱滤波器系数可以使得卷积核在空间上局部化，使用了这个形式：

$$\tag{3}
\hat{g}\_i = \sum^r\_{j=1} \alpha\_i \beta\_j (\Lambda\_i)
$$

其中 $\beta\_1(\lambda), \dots, \beta\_r(\lambda)$ 是一些固定的插值核，$\mathbb{\alpha} = (\alpha\_1, \dots, \alpha\_r)$ 是插值系数。矩阵形式中，滤波器写为 $\text{diag}(\hat{G}) = \mathbf{B\alpha}$，其中 $\bf{B} = (b\_{ij}) = (\beta\_j (\lambda\_i))$ 是一个 $k \times r$ 的矩阵。这样一个参数化可以使参数保持在 $n$ 个。

**Chebyshev Spectral CNN (ChebNet).** 为了减轻计算图傅里叶变换的代价，Defferrard et al 使用了切比雪夫多项式来表示卷积核：

$$\tag{4}
g\_\alpha(\Delta) = \sum^{r-1}\_{j=0} \alpha\_j T\_j(\tilde{\Delta}) = \sum^{r-1}\_{j=0} \alpha\_j \Phi T\_j (\tilde{\Lambda}) \Phi^T,
$$

其中 $\tilde{\Delta} = 2 \lambda^{-1}\_n \Delta - \bf I$ 是 rescaled 拉普拉斯矩阵，它的特征值 $\tilde{\Lambda} = 2 \lambda^{-1}\_n \Lambda - \bf I$ 在区间 $[-1, 1]$ 内，$\alpha$ 是 $r$ 维的滤波器中的多项式系数，

$$\tag{5}
T\_j(\lambda) = 2 \lambda T\_{j-1}(\lambda) - T\_{j-2} (\lambda),
$$

表示 $j$ 阶切比雪夫多项式，$T\_1(\lambda) = \lambda$，$T\_0(\lambda) = 1$。

这样的方法有几个优点。首先，它不需要计算拉普拉斯矩阵的特征向量。由于切比雪夫多项式的递归定义，计算滤波器 $g\_\alpha(\Lambda) \bf f$ 要使用拉普拉斯矩阵 $r$ 次，会导致一个 $\mathcal{O}(rn)$ 的操作。其次，因为拉普拉斯矩阵是一个局部操作，只影响顶点的一阶邻居，它的 $(r-1)$次幂影响 $r$阶邻居，得到的滤波器是局部化的。