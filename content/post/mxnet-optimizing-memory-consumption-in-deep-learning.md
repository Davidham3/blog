---
categories:
- software
date: 2019-04-07 13:02:17+0000
description: '[MXNet: Optimizing Memory Consumption in Deep Learning](https://mxnet.incubator.apache.org/versions/master/architecture/note_memory.html)'
draft: false
math: null
tags:
- software
title: 'MXNet: Optimizing Memory Consumption in Deep Learning'
---
[MXNet: Optimizing Memory Consumption in Deep Learning](https://mxnet.incubator.apache.org/versions/master/architecture/note_memory.html)
<!--more-->

过去的十年，深度学习模型趋向于更深更大的网络。尽管硬件性能的快速发展，前沿的深度学习模型还是继续将 GPU 显存的限制推向极限。所以即便在今天，人们仍在寻找办法消耗更少的内存，训练更大的模型。这样可以让我们训练的更快、使用更大的批量、获得更高的 GPU 利用率。

在这篇文档中，我们讨论集中优化内存分配的技术。尽管我们的讨论不彻底，但这些方案具有指导意义，使我们能够介绍主要的设计问题。

# Computation Graph

计算图描述了操作间的依赖。图中的操作要么是细粒度的，要么是粗粒度的。下图展示了两个计算图的例子。

![Figure1](/images/mxnet-optimizing-memory-consumption-in-deep-learning/Fig1.png)

计算图的概念被明确地编码进了库中，如 Theano 和 CGT。其他库中，计算图隐式地作为网络的配置文件。主要区别是如何计算梯度。主要有两种方法：在同一个图上做反向传播或明确地表示出一个回溯的路径来计算需要的梯度。

![Figure2](/images/mxnet-optimizing-memory-consumption-in-deep-learning/Fig2.png)

像 Caffe，CXXNet，Torch这样的框架使用前者，在原图上做反向传播。Theano 和 CGT 使用后者，显示地表示反向路径。我们讨论显示地反向路径方法，因为它对于优化有几个优势。

然而，我们应该强调一下选择显示反向路径方法并不会限制我们使用符号式的库，如 Theano 和 CGT。我们也可以用显示反向路径对基于层（将前向和反向绑起来）的库进行梯度计算。下面的图表示了这个过程。基本上来说，我们引入反向结点，连接图中的前向节点，在反向操作的时候调用 `layer.backward`。

![Figure3](/images/mxnet-optimizing-memory-consumption-in-deep-learning/Fig3.png)

这个讨论可以应用在几乎所有现存的深度学习框架上。

为什么显示反向路径更好？我们可以看两个例子。第一个原因是显示反向路径清晰地描述了计算间的依赖关系。考虑一种情况，我们想获得 A 和 B 的梯度。我们可以从图中清楚地看到，`d(C)` 梯度的计算不依赖于 F。这意味着我们可以在前向传播完成后释放 `F` 的内存。类似的，`C` 的内存也可以被回收。

![Figure4](/images/mxnet-optimizing-memory-consumption-in-deep-learning/Fig4.png)

拥有不同的反向路径而不是前向传播的镜像的能力是其另一个优点。一个常见的例子是分离连接的情况，如下图：

![Figure5](/images/mxnet-optimizing-memory-consumption-in-deep-learning/Fig5.png)

在这个例子中，B 的输出由两个操作引用。如果我们想在同一个网络中计算梯度，我们需要引入一个显示的分割层。这意味着我们需要对前向也做一次分离。如图，前向不包含一个分割层，但是图会自动地在将梯度传回 B 之前插入一个梯度聚合结点。这有助于我们节省分配分割输出的内存成本，以及在前向传递中复制数据的操作成本。

如果我们应用显示反向方法，在前向和反向的时候就没有区别。我们简单地按时间顺序进入计算图，开始计算。这使得显示反向路径容易去分析。我们仅需要回答一个问题：我们如何对计算图每个输出结点分配内存？

# What Can Be Optimized?

计算图是一种讨论内存分配优化技术有用的方式。我们已经想你展示了如何通过显示反向图节省内存。现在我们讨论些进一步的优化，看看如何确定基准测试的合理基线。

假设我们想构建 `n` 层神经网络。一般来说，在我们实现神经网络的时候，我们需要同时为每层的输出和反向传播时的梯度分配空间。这意味着我们需要差不多 `2n` 的内存。在显示反向图方法中我们面对的是同样的需求因为反向传播时结点数与前向传播差不多。

## In-place Operations

我们可以使用的一个最简单的技术就是跨操作的原地内存共享。对于神经网络，我们通常将这个技术应用在对应操作的激活函数上。考虑下面的情况，我们想计算三个链式 sigmoid 函数的值：

![Figure6](/images/mxnet-optimizing-memory-consumption-in-deep-learning/Fig6.png)

因为我们可以原地计算 sigmoid，使用同样的内存给输入和输出，我们可以使用固定的内存大小计算任意长度的链式 sigmoid 函数。

注意：在实现原地优化时很容易犯错误。考虑下面的情况，B 的值不仅用于 C，还用于 F。

![Figure7](/images/mxnet-optimizing-memory-consumption-in-deep-learning/Fig7.png)

我们不能使用原地优化因为 B 的值在 `C = sigmoid(B)` 计算之后仍然需要。如果一个算法简单地对所有 sigmoid 函数都做这个原地优化就会掉进这个陷阱，所以在使用的时候，我们需要注意这个问题。

## Standard Memory Sharing

除了原地操作还有其他地方可以共享内存。下面的例子中，因为 B 的值在计算 E 之后不再需要，我们可以重新使用 B 的内存来存储 E。

![Figure8](/images/mxnet-optimizing-memory-consumption-in-deep-learning/Fig8.png)

内存共享不需要相同大小的数据。注意再上面的例子中，B 和 E 的 shape 可以不一样。为了处理这样的情况，我们可以分配一个等价于 B 和 E 中大的那个元素的大小，然后让他们共享这个区域。