---
date: 2022-05-23 15:27:10+0000
draft: false
math: true
tags:
- deep learning
- point process
title: Semi-supervised Learning for Marked Temporal Point Processes
---

[Semi-supervised Learning for Marked Temporal Point Processes](https://arxiv.org/pdf/2107.07729.pdf)。MTPP的半监督学习，模型称为SSL-MTPP。有标签的地方就用RMTPP，没有标签的地方用RMTPP的编码器和解码器来重构。两边的损失加在一起优化网络。

<!--more-->

![Figure1](/images/semi-supervised-learning-for-marked-temporal-point-processes/Fig1.jpg)

# 3 Proposed Algorithm

![Figure2](/images/semi-supervised-learning-for-marked-temporal-point-processes/Fig2.jpg)

架构如图2所示，损失函数：

$$
\tag{1} \mathcal{L}\_{SSL-MTPP} = \mathcal{L}\_{Time} + \mathcal{L}\_{Marker} + \mathcal{L}\_{Recon}
$$

## 3.1 SSL-MTPP Algorithm

有标签的数据，一组序列$(S)$，包含$n$个序列pair，$(x\_i, y\_i)$，$(x\_i)$是事件的时间信息，$(y\_i)$是marker信息。用RNN捕获marker和序列的时间信息。嵌入表示用于预测marker和时间。

没有标签的数据，用RNN编解码器模型，只学习时间信息。学习到的时间表示用来增强marker-time embedding。

**Unsupervised Reconstruction Loss Component**

重构损失，只重构时间，不考虑marker，因此有没有标签都可以用。给定$n$个序列的训练集$S = \{x\_1, x\_2, \dots, x\_n \}$，每个序列$x\_i$包含$k$个事件，重构损失定义为：

$$
\tag{2} \mathcal{L}\_{Recon} = \sum^n\_{i=1} \Vert x\_i - \mathcal{D}(\mathcal{E}(x\_i)) \Vert^2\_2
$$

$\mathcal{E}$和$\mathcal{D}$分别表示RNN编码器和RNN解码器。重构损失专注于在给定的时间序列上学习有意义的表示，用于后续marker的预测。重构损失在训练过程完全是无监督的。$(\mathcal{E}(x\_i))$是时间序列的编码。如何用这个嵌入表示预测后续的marker后面会讲。

![Figure3](/images/semi-supervised-learning-for-marked-temporal-point-processes/Fig3.jpg)

**Supervised Marker and Time Prediction Loss Components**

$(x\_i, y\_i)$包含事件的时间信息和marker信息，输入到RNN模块后可以获得marker和时间相互依赖的表示：

$$
\tag{3} f\_i = RNN(x\_i, y\_i)
$$

提取出的特征表示与无监督的时间表示$(\mathcal{E}(x\_i))$一起生成融合嵌入表示：

$$
\tag{4} f^{fused}\_i = f\_i + \lambda \ast \mathcal{E}(x\_i)
$$

$\lambda$是权重。这个融合表示放入一个2层感知机预测下一个事件的时间和marker。预测模型通过下面的损失来训练：

$$
\tag{5} \begin{align} \mathcal{L}\_{Marker} &= - \sum^M\_{c=1} y^i\_{i,c} \log(p^j\_{i,c})\\ \mathcal{L}\_{Time} &= \Vert x^j\_i - {x^j}'\_i \Vert \end{align}
$$

$\mathcal{L}\_{Marker}$用交叉熵，$\mathcal{L}\_{Time}$用MAE损失。事件$j$是序列$i$的一个事件，时间是$x^j\_i$，marker是$y^j\_i$，预测的类别有$M$个。$y^j\_{i,c}$是一个binary变量，表示样本$y^j\_i$是否是类别$c$，$p^j\_{i,c}$是样本属于类别$c$的概率，${x^j}’\_i$是给定事件的预测时间。

## 3.2 Implementation Details

SSL-MTPP利用了RMTPP的架构。监督部分的RNN是一个5层LSTM模型，无监督部分是2层的RNN编码器和解码器。marker和event prediction模块分别用了2个dense层。RNN后面用了Dropout。$\lambda$设为0.1。Adam，学习率0.01，训练100轮，batch size是1024个sequence。