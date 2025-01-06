---
categories:
- 论文阅读笔记
date: 2018-12-03 15:17:12+0000
description: Layer Normalization，之前看到一篇论文用了这个LN层，看一下这个怎么实现。原文链接：[Layer Normalization](https://arxiv.org/abs/1607.06450.pdf)
draft: false
math: true
tags:
- deep learning
- machine learning
title: Layer Normalization
---
Layer Normalization，之前看到一篇论文用了这个LN层，看一下这个怎么实现。原文链接：[Layer Normalization](https://arxiv.org/abs/1607.06450.pdf)
<!--more-->

# Abstract

训练神经网络很费时，一个减少训练时间的方法是对神经元的激活值归一化。最近的一项技术称为小批量归一化，也就是 batch norm，使用一个神经元的输入的分布，在一个批量的样本上计算均值和方差，然后在神经元的每个训练样例上做归一化。这个能极大地缩短训练时间。但是，batch norm 的效果和 batch size 有关，而且还不知道怎么应用在 RNN 上。我们使用一个训练样例，将 BN 转置，计算一个层上面所有神经元的输入的均值和方差来归一化。就像 BN 一样，我们在归一化后激活之前给每个神经元它自己的可适应的bias和gain。不像 BN 的地方是，LN 在训练和测试的时候都有，通过在每个时间步上做归一化的统计，LN 也能应用在 RNN 上。LN 在稳定 RNN 隐藏状态的动态性上面很有效。经验表明，LN 与之前的技术对比能有效地减少训练时间。

# 1 Introduction

很多深度神经网络要训练好多天。BN 除了提升了收敛速度，从批量统计量得到的随机性在训练的时候还会作为一个正则项。

尽管 BN 简单，但是它需要输入统计量之和的平均值。在定长的 FNN 中，把每个层的 BN 存起来就行。但是，RNN 的循环单元的输入通常随序列长度而变化，所以将 BN 应用在 RNN 上面，不同时间步需要不同的统计量。此外，BN 不能应用在在线学习等任务上，或是非常大的分布式模型上，因为 minibatch 会很小。

# 2 Background

前向神经网络是从输入模式 $\rm{x}$ 映射到输出向量 $y$ 的非线性变换。在深度前向神经网络中的第 $l$ 个隐藏层，$a^l$ 表示这层神经元的输入。汇总后的输入通过一个线性映射计算如下：

$$\tag{1}
a^l\_i = {w^l\_i}^\text{T} h^l\\
h^{l+1}\_i = f(a^l\_i + b^l\_i)
$$

其中 $f(\cdot)$ 是激活函数，$w^l\_i$ 和 $b^l\_i$ 分别是第 $l$ 个隐藏层的权重和偏置参数。参数通过基于梯度的学习方法得到。

深度学习的一个挑战是：某一层权重的梯度和上一层的输出高度相关，尤其是当这些输出以一种高度相关的方式变化的时候。BN 提出来是减少这种不希望的 covariate shift 现象。这种方法在输入样例在每个隐藏单元的输入上做计算。详细来说，对于第 $l$ 层的第 $i$ 个输入，BN 根据他们在数据中的分布，将输入缩放了：

$$\tag{2}
\bar{a}^l\_i = \frac{g^l\_i}{\sigma^l\_i}(a^l\_i - \mu^l\_i)\\
\mu^l\_i = \mathbb{E}\_{\mathrm{x} \sim P(\mathrm{x})}[a^l\_i]\\
\sigma^l\_i = \sqrt{\mathbb{E}\_{\mathrm{x} \sim P(\mathrm{x})}[(a^l\_i - \mu^l\_i)^2]}
$$

其中 $\bar{a}^l\_i$ 是第 $l$ 层第 $i$ 个输入的归一化结果，$g\_i$ 是在非线性激活函数之前的一个增益参数，对归一化激活值进行缩放。注意，期望是在所有训练数据上的。事实上计算式2中的期望是不实际的，因为这需要用当前的参数，前向传播过所有的训练集。实际中是用当前的 mini-batch 来估计 $\mu$ 和 $\sigma$。这就给 batch size 增加了限制，而且很难应用到 RNN 上。

# 3 Layer normalization

层归一化用来克服批量归一化的一些缺点。

一个层输出的变换倾向于导致下一层的输入之间有着关联度很高的变化，尤其是使用 ReLU 激活后，这些输出的变化很多。这表明 covariate shift 问题可以通过固定每层的输入的均值和方差解决。因此，在同一层中所有隐藏单元的层归一化统计量如下：

$$\tag{3}
\mu^l = \frac{1}{H} \sum^H\_{i = 1}a^l\_i\\
\sigma^l = \sqrt{\frac{1}{H} \sum^H\_{i=1} (a^l\_i - \mu^l)^2}
$$

其中 $H$ 表示层内的隐藏单元数。式2和式3的区别是在层归一化之下，层内所有隐藏单元共享相同的归一化项 $\mu$ 和 $\sigma$，但是不同的样本有着不同的归一化项。不像 BN，层归一化不会有 batch size 的限制，而且可以使用在 batch size 设为1的时候的在线学习上。

## 3.1 Layer normalized recurrent neural networks

最近的序列到序列模型 [Sutskever et al., 2014] 利用了紧致的 RNN 来解决 NLP 中的序列预测问题。在 NLP 任务中不同的训练样例长度不一致是很常见的。RNN 在每个时间步使用的参数都是相同的。但是在使用 BN 来处理 RNN 时，我们需要计算并存储序列中每个时间步的统计量。如果一个测试的序列比任何训练的序列都长，那就会出问题了。层归一化不会有这样的问题，因为它的归一化项只依赖于当前时间步层的输入。它在所有的时间步上也有一组共享的 gain 和 bias 参数。

在标准的 RNN 中，循环层的输入通过当前的输入 $\mathrm{x}^t$ 和前一层的隐藏状态 $\mathrm{h}^{t-1}$，得到 $\mathrm{a}^t = W\_{hh}h^{t-1} + W\_{xh} \mathrm{x}^t$。层归一化后的循环层会将它的激活值使用像式3一样的归一化项缩放到：

$$\tag{4}
\mathrm{h}^t = f[\frac{\mathrm{g}}{\sigma^t} \odot (\mathrm{a}^t - \mu^t) + b]\\
\mu^t = \frac{1}{H} \sum^H\_{i=1}a^t\_i\\
\sigma^t = \sqrt{\frac{1}{H} \sum^H\_{i=1}(a^t\_i - \mu^t)^2}
$$

其中 $W\_{hh}$ 是循环隐藏到隐藏的权重，$W\_{xh}$ 是输入到隐藏的权重，$\odot$ 是element-wise multiplication。$\rm b$ 和 $\rm g$是和 $\mathrm{h}^t$ 同维度的 bias 和 gain 参数。

在标准的 RNN 中，每个时间步的循环单元的输入的数量级倾向于增大或减小，导致梯度的爆炸或消失问题。在一个层归一化的 RNN 里，归一化项使它对一个层的输入的缩放不发生变化，使得隐藏到隐藏动态性更稳定。