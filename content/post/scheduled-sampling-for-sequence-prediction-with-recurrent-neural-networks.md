---
categories:
- 论文阅读笔记
date: 2018-08-11 10:14:13+0000
description: NIPS 2015. 在训练seq2seq的时候，比如像机器翻译，训练的时候，每个输出y，它所依据的前一个词，都是正确的。但是在预测的时候，输出的这个词依照的上一个词，是模型输出的词，无法保证是正确的，这就会造成模型的输入和预测的分布不一致，可能会造成错误的累积。本文提出了scheduled
  sampling来处理这个问题。原文链接：[Scheduled Sampling for Sequence Prediction with Recurrent
  Neural Networks](https://arxiv.org/abs/1506.03099)
draft: false
math: true
tags:
- deep learning
- Sequence
title: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks
---
NIPS 2015. 在训练seq2seq的时候，比如像机器翻译，训练的时候，每个输出y，它所依据的前一个词，都是正确的。但是在预测的时候，输出的这个词依照的上一个词，是模型输出的词，无法保证是正确的，这就会造成模型的输入和预测的分布不一致，可能会造成错误的累积。本文提出了scheduled sampling来处理这个问题。原文链接：[Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/abs/1506.03099)

<!--more-->

# Abstract

循环神经网络可以训练成给定一些输入，输出一些token的模型，比如最近的机器翻译和图像描述。现在训练这些的方法包括给定当前状态和之前的token，最大化序列中每个token的似然。在推理阶段，未知的token会被模型生成的token替代。这种在训练和推断之间的差异，会在生成序列时产生快速积累的误差。我们提出了一个递进学习策略(curriculum learning strategy)轻微地将训练过程进行一些改变，从之前的完全使用一个真实的token进行引导的策略，变成了基本使用生成的token来引导的策略。在几个序列预测的实验上表明我们的方法有很大的提升。此外，它成功的在MSCOCO图像描述2015任务上获得冠军。

# 1 Introduction

循环神经网络可以用于处理序列，要么输入，要么输出，或者都可以。尽管他们很难在长期依赖的数据上训练，一些如LSTM的版本可以更好的适应这种问题。事实是，最近在一些序列预测问题上，包括机器翻译，contextual parsing，图像描述甚至视频描述上，这些模型表现的很好。

在这篇论文中，我们考虑生成可变大小的token的序列的问题，比如机器翻译，目标是给定源语言翻译成目标语言。我们也考虑当输入不是一个序列的问题，如图像描述问题，目标是对给定的图像生成一个文本描述。

在这两种情况，循环神经网络一般都是通过给定的输入，最大化生成目标序列的似然。实际上，这个是通过给定当前模型的状态和之前的目标token，最大化每个目标token的似然，这使得模型可以在目标token上学习一种语言模型。然而，在推断过程中，真的*previous*目标token是无法获取的，这就使得模型需要用它自己生成的token，导致模型在训练和预测时会有差异。通过使用beam search启发式的生成几个目标序列可以缓解这种差异，但是对于连续的状态空间模型，如RNN，不存在动态规划的方法，所以即便是使用beam search，考虑的序列的数量仍然会很小。

主要问题是，在生成序列时越早出现错误，会导致将这个错误输入进模型，然后会扩大模型的误差，因为模型会将它在训练时未见过的错误考虑到状态空间内。

我们提出了一个递进学习方法，在对序列预测任务上使用RNN的训练和推测时构建了桥梁。我们提出，改变训练过程，为了逐渐地使模型处理它的错误，使它在推断时也可以进行。这样，模型在训练时会探索更多的情况，因此在推断时会更鲁棒的纠正它的错误，因为它在训练时就学习过这个。我们会展示这个方法在几个序列预测问题上的结果。

# 2 Proposed Approach

我们考虑一个监督学习任务，训练集是给定的 $N$ 个样本的输入输出对，$\lbrace X^i, Y^i \rbrace^N\_{i=1}$，$X^i$ 是输入，要么静态（图像），要么动态（序列），输出 $Y^i$ 是一个可变数量的token的序列 $y^i\_1, y^i\_2, ..., y^i\_{T\_i}$，token属于一个已知的词典。

## 2.1 Model

给定一个输入/输出对儿 $(X, Y)$，log 概率 $P(Y \mid X)$ 可由下式计算：

$$\tag{1}
\begin{aligned}
\mathrm{log} P(Y \mid X) &= \mathrm{log} P(y^T\_1 \mid X) \\
&= \sum^T\_{t = 1} \mathrm{log} P(y\_t \mid y^{t-1}\_1, X)
\end{aligned}
$$

其中，$Y$ 是长度为 $T$ 的序列，$y\_1, y\_2, ..., y\_T$。在前面的等式中，后面的项通过一个参数为 $\theta$ 的循环神经网络，通过一个状态向量 $h\_t$估计得到，也就是通过前一个输出 $y\_{t-1}$ 和前一个状态 $h\_{t-1}$估计得到：

$$\tag{2}
\mathrm{log} P(y\_t \mid y^{t-1}\_1, X; \theta) = \mathrm{log} P(y\_t \mid h\_t; \theta)
$$

其中，$h\_t$ 通过如下的一个循环神经网络计算得到：

$$\tag{3}
h\_t = \begin{cases}
f(X; \theta) \ \ \mathrm{if} t = 1\\
f(h\_{t-1}, y\_{t-1}; \theta) \mathrm{otherwise}.
\end{cases}
$$

$P(y\_t \mid h\_t; \theta)$ 经常通过状态向量 $h\_t$ 的一个线性变换，变换到一个 vector of scores 实现，这个向量是输出字典的每个token的分数，然后用一个 softmax 确保分数适当的归一化。$f(h, y)$通常是一个非线性函数，这个函数融合了之前的状态和之前的输出来生成当前的状态。

这就意味着模型专注于给定模型当前状态，学习预测下一个输出。因此，模型会以最普通的形式表示序列的概率分布——不像条件随机场以及其他的模型，在给定隐变量状态后，假设不同时间步的输出相互独立。模型的容量只会被循环层和前向传播层的表示容量限制。LSTM，因为他们能学习长范围的结构，所以对这种问题来说非常适合，也就可以学习序列上的rich distributions。

为了学习边长序列，一个特殊的token，<EOS>，表示序列的结束被添加进字典和模型中。在训练的过程中，<EOS>会拼接在每个序列的结尾处。在推理的时候，模型会生成tokens直到它生成了<EOS>。

## 2.2 Training

训练循环神经网络来解决这样的问题通常通过mini-batch随机梯度下降求解，通过给定输入数据 $X^i$，为所有的训练对儿 $(X^i, Y^i)$最大化生成正确的目标序列 $Y^i$ 的似然，找到一组参数$\theta^*$。

$$\tag{4}
\theta^* = \mathop{\arg \max\_\theta} \sum\_{(X^i, Y^i)} \mathrm{log} P(Y^i \mid X^i; \theta)
$$

## 2.3 Inference

在推理的过程中，模型可以在给定 $X$ 的情况下通过一次生成一个token，生成整个序列 $y^T\_1$。生成一个 <EOS> 后，它标志着序列的结束。对于这个过程，在时间 $t$，模型为了生成 $y\_t$，需要从最后一个时间戳讲输出的token $y\_{t-1}$ 作为输入。因为我们没法知道真正的上一个token是什么，我们可以要么选择模型给出的最可能的那个，要么根据这个来抽样。

给定 $X$ 搜索最大概率的序列 $Y$ 非常费时，因为序列的长度是组合地上升的。我们使用一个beam search来生成 $k$ 个最好的序列。我们通过维护 $m$ 个最优候选序列组成的一个堆来做这个。每次通过给每个候选序列扩充一个token并把它增加到堆内，都能得到一个新的候选序列。在这步的结尾，堆重新地剪枝到只有 $m$ 个候选序列。beam searching在没有新的序列增加的时候就会截断，然后返回 $k$ 个最优的序列。

尽管beam search通常用于基于HMM这样的模型的离散状态，这些模型可以使用动态规划，但是对于像RNN这样的连续状态模型就很难了，因为没有办法再连续空间内factor the followed state paths，因此在beam search解码的时候，可以控制的候选序列的实际数量是很小的。

在所有的情况里，如果在时间 $t-1$ 有一个错误生成，那么模型就会在一个和训练分布不同的状态空间中，而且它会在这个空间中不知所措。更糟的是，这会导致模型在决策的时候导致不好的决策的累计——a classic problem in sequential Gibbs sampling type appraoches to sampling, where future samples can have no influence on the past.

## 2.4 Bridging the Gap with Scheduled Sampling

在预测token $y\_t$ 时，训练和推断的主要差别是我们是否使用前一个真实的token $y\_{t-1}$，还是使用一个从模型得到的估计值 $\hat{y}\_{t-1}$。

我们在这里提出了一个采样机制，会在训练的时候，随机地选择 $y\_{t-1}$ 或 $\hat{y}\_{t-1}$。假设我们使用mini-batch随机梯度下降，对于训练算法的第 $i$ 个mini-batch中预测 $y\_t \in Y$ 的每个token，我们提出用抛硬币的方法，设使用真实token的概率为$\epsilon\_i$，使用它估计的token的概率为$(1 - \epsilon\_i)$。模型的估计值可以根据模型的概率分布$P(y\_{t-1} \mid h\_{t-1})$ 来采样获得，或是取 $\mathop{\arg \max\_s} P(y\_{t-1} = s \mid h\_{t-1})$。这个过程由图1所示。

<div align=center>![Figure1](/blog/images/scheduled-sampling-for-sequence-prediction-with-recurrent-neural-networks/Fig1.jpg)

当 $\epsilon\_i = 1$ 时，模型就像之前一样训练，但是当 $\epsilon\_i = 0$ 时，模型就会和推断时一样训练。我们这里提出了一个递进学习策略，在训练的开始接断，从模型可能生成的token中进行采样，因为此时模型还没有训练好，这可能会使模型的收敛速度变慢，所以这里选择较多的真实token会帮助训练；另一方面，在训练快结束的时候，$\epsilon\_i$ 应该更倾向于从模型的生成结果中采样，因为这个对应了推测的场景，这时我们会期望模型已经有足够好的能力来处理这个问题，并且采样出有效的tokens。

因此我们提出使用一个规则来减少 $\epsilon\_i$ 来作为 $i$ 的函数，就像现在很多随机梯度下降算法那样降低学习率一样。这样的规则如图2所示：

· Linear decay: $\epsilon\_i = \mathrm{max} (\epsilon, k - ci)$，其中 $0 \leq \epsilon < 1$ 是给模型的真值最小的数量，$k$ 和 $c$ 提供了衰减的截距和斜率，这些依赖于收敛速度。

· Exponential decay: $\epsilon\_i = k^i$，其中 $k < 1$ 是一个依赖于期望收敛速度的常量。

· Inverse sigmoid decay: $\epsilon\_i = k/(k + \mathrm{exp}(i/k))$，其中 $k \geq 1$ 依赖于期望收敛速度。

我们的方法命名为 *Scheduled Sampling*。需要注意的是在模型训练时从它的输出采样到前一个token $\hat{y}\_{t-1}$时，我们可以在时间 $t \rightarrow T$ 内进行梯度的反向传播。在实验中我们没有尝试，会在未来的工作中尝试。

<div align=center>![Figure2](/blog/images/scheduled-sampling-for-sequence-prediction-with-recurrent-neural-networks/Fig2.jpg)