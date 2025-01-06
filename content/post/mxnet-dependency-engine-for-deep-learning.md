---
categories:
- software
date: 2019-04-07 12:18:41+0000
description: '[MXNet: Dependency Engine for Deep Learning](https://mxnet.incubator.apache.org/versions/master/architecture/note_engine.html)'
draft: false
math: null
tags:
- software
title: 'MXNet: Dependency Engine for Deep Learning'
---

[MXNet: Dependency Engine for Deep Learning](https://mxnet.incubator.apache.org/versions/master/architecture/note_engine.html)
<!--more-->

我们总是想让深度学习框架跑的更快，能适应更大的数据集。一个自然的想法是我们能否通过堆更多的硬件解决问题，也就是同时使用多个 GPU。

框架设计者就会问：我们怎么才能让计算在设备间并行？而且，当我们引入多线程的时候，如何同步计算？一个运行环境依赖引擎是这些问题的解决方案。

在这篇文档中，我们检验了使用运行环境依赖调度装置来加速深度学习的方法，解释了运行环境依赖调度器如何同时加速和简化多设备深度学习。我们还探索了框架独立或操作独立的通用依赖引擎可能的设计方案。

这里的很多讨论都是源于 MXNet 依赖引擎。我们讨论的依赖追踪算法主要由 [Yutian Li](https://github.com/hotpxl) 和 [Mingjie Wang](https://github.com/jermainewang) 设计。

# Dependency Scheduling

尽管大多数用户想利用并行计算，但大部分人更熟悉串行编程。所以一个问题是：我们如何能写串行程序，构建一个库，自动地并行我们的程序？

举个例子，下面的代码，我们可以以任意顺序运行 `B = A + 1` 和 `C = A + 2` 这两个命令，或是并行运行：

```
A = 2
B = A + 1
C = A + 2
D = B * C
```

但是由于最后一个操作 `D = B * C`，导致手动编码序列很麻烦，最后一个操作需要等待前面的操作完成才能继续。下面的依赖图/数据流图展示了这个过程。

![Figure1](/images/mxnet-dependency-engine-for-deep-learning/Fig1.png)

一个依赖引擎可以获取一个操作序列并且根据依赖关系调度他们，更可能以并行的方式。所以在这个例子中，一个依赖库可以并行运算 `B = A + 1` 和 `C = A + 2`，然后在这两个操作完成后运行 `D = B * C`。

# Problems in Dependency Scheduling

一个依赖引擎减轻了编写并发程序的负担。但是，由于操作可以并行化，新的依赖追踪问题产生了，这节我们讨论这些问题。

## Data Flow Dependency

数据流依赖表述了一个计算的输出如何用于其他的计算。每个依赖引擎必须解决数据流依赖问题。

![Figure2](/images/mxnet-dependency-engine-for-deep-learning/Fig2.png)

因为我们在前面的部分讨论过这个问题，我们这里使用同一张图。包含数据流追踪引擎的框架包括 Minerva 和 Purine2。

## Memory Recycling

我们什么时候回收分配给 array 的内存？在串行程序中这个问题很简单。我们在变量在作用域中消失后回收即可。但是，下面的图展示了并行程序中这有多麻烦。

![Figure3](/images/mxnet-dependency-engine-for-deep-learning/Fig3.png)

在这个例子中，两个操作都需要 `A` 的值，我们需要等两个操作都完成才能回收。引擎必须根据依赖来调度回收器，确保在 `B = A + 1` 和 `C = A + 2` 都完成后再执行。

## Random Number Generation

随机数生成器是机器学习中常用的，给依赖引擎提出了有趣的挑战。考虑下面的问题：

![Figure4](/images/mxnet-dependency-engine-for-deep-learning/Fig4.png)

再这个例子中，我们以序列形式生成了随机数。尽管看起来两个随机数生成过程是并行的，但实际上不是。一个伪随机数生成器 (PRNG) 不是线程安全的，因为在生成新的随机数时，可能会导致一些内部状态的变化。即使 PRNG 是线程安全的，我们也希望数字的生成是串行的，因为我们可以得到可重现的随机数序列。