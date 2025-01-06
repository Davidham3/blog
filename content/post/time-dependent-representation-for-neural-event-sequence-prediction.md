---
date: 2022-04-27 15:37:58+0000
description: ICLR 2018 workshop, [Time-dependent representation for neural event sequence
  prediction](https://arxiv.org/abs/1708.00065)。事件序列的表示学习模型。主要是对事件的嵌入表示有了一些创新，加入了对事件duration的考虑。模型整体还是RNN架构。
draft: false
math: true
tags:
- deep learning
- event sequence
- learning representations
title: Time-dependent representation for neural event sequence prediction
---

ICLR 2018 workshop, [Time-dependent representation for neural event sequence prediction](https://arxiv.org/abs/1708.00065)。事件序列的表示学习模型。主要是对事件的嵌入表示有了一些创新，加入了对事件duration的考虑。模型整体还是RNN架构。

<!--more-->

![Figure1](/images/time-dependent-representation-for-neural-event-sequence-prediction/Fig1.jpg)

在事件序列中，有两个时间段(time span)，一个是duration，事件的持续时长，另一个是interval，事件与事件之间的间隔。为了统一这两个时间段，作者将interval看作是一个空闲事件(idle event)。

![Figure2](/images/time-dependent-representation-for-neural-event-sequence-prediction/Fig2.jpg)

图2是模型架构，就是把事件做嵌入表示，然后把duration考虑进去，得到事件序列里面每个时间步的嵌入表示，然后丢到RNN里面，最后是要预测下一个event是什么，同时可以把下一个event的duration拿进来计算损失，起到一个正则的作用。

## 3.1 Contextualizing Event Embedding With Time Mask

这里有一个观点，就是在机器翻译中，RNN会花一定的capacity去区分不同context下一个词的意思。为了解决这个问题，Choi et al., 2016 提出了一个mask计算。本文借助这个思想，提出了一个时间依赖的嵌入表示。

$$
\tag{1} c^d = \phi (\log(d\_t);\theta)
$$

上式是一个FNN，$\phi$是非线性变换，参数是$\theta$。加对数是为了降低$d\_t$的范围。然后用一个单层sigmoid的非线性变换把$c^d$映射到$m^d \in \mathbb{R}^E$上，这个就是mask。

$$
\tag{2} m\_d = \sigma(c^d W\_d + b\_d)
$$

然后

$$
\tag{3} x\_t \leftarrow x\_t \odot m\_d
$$

把mask和一个事件的embedding相乘，element-wise production，这个东西放入RNN。

## 3.2 Event-Time Joint Embedding

这里有个观点，我们平时只会说“和谁简单地聊了一会儿”，不会说具体聊了多少分钟。我们对于事件时长的感受依赖事件的类型。基于这个直觉，我们用了一个sort one-hot嵌入表示，做了一个事件的联合表示。

首先把事件时长映射到一个向量上去

$$
\tag{4} p^d = d\_t W\_d + b\_d \in \mathbb{R}^P
$$

然后在这个向量上面做一个softmax运算

$$
\tag{5} s^d\_i = \frac{\exp(p^d\_i)}{\sum^P\_{k=1} \exp(p^d\_k)}
$$

然后做一个线性变换

$$
\tag{6} g\_d = s^d E^s \in \mathbb{R}^E
$$

然后把事件嵌入和这个时间嵌入求平均，得到事件嵌入表示：

$$
\tag{7} x\_t \leftarrow \frac{x\_t + g\_d}{2} \in \mathbb{R}^E
$$


# 4 Next Event Duration as A Regularizer

这里讨论的是通过让模型去预测下一事件的时长来增强模型。这个时长是通过对RNN循环层做线性变换得到的。对于时间步$t$，来说，需要预测的duration是$d’\_{t+1}$。它的损失回传后会起到一个正则的作用。而且可以对事件预测输出层路径上的多个层进行正则。

## 4.1 Negative Log Lieklihood of Time Prediction Error

这里说，对于连续值的预测，一般用MSE，但是MSE这个指标需要和事件预测的损失在同一个数量级上。而事件损失，一般是一个log形式的损失，也就是说这个数会比较小。Hinton & van Camp, 1993研究证明最小化平方损失可以写成最大化0均值高斯分布的概率密度，而且不需要duration服从高斯分布，但是预测误差需要。因此正则项要做一个标准化，

$$
\tag{8} R^N\_t = \frac{(d'\_{t+1} - d\_{t+1})^2}{2\sigma^2\_i}
$$

$\sigma\_i$是通过训练集的duration算出来的，然后在训练的过程中，通过时长预测误差的分布来更新。

## 4.2 Cross Entropy Loss on Time Projection

这里说，对于时长的损失计算还可以用softmax。

因为3.2节提到了一个把连续值映射到向量空间的办法，使用同样的办法可以计算另一种损失：

$$
\tag{9} R^X\_t = - \sum^P\_{k=1} Proj\_k (d\_{t+1}) \log{Proj\_k (d'\_{t+1})}
$$

$Proj$就是公式4和5定义的投影函数，$Proj\_k$是投影向量中的第$k$项。当3.2节的事件与时间的联合嵌入表示和这个损失都使用的时候，可以把投影函数的权重共享。

# 5 Experiments

用了5个数据集。

## 5.1 数据预处理

做了一些特别稀有的事件的过滤。有些事件少于5次的用OOV代替了。使用MAP@K和Precision@K来评估。

训练、验证、测试的比例是8:1:1

## 5.2 模型配置

* NoTime: 就用一个简单的LSTM
* TimeConcat: 把duration做log变换，与事件嵌入表示拼接，输入RNN
* TimeMask: 3.1节的方法
* TimeJoint: 3.2节的方法
* RMTPP: [RMTPP](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf)

## 5.4 实验结果

![Figure3](/images/time-dependent-representation-for-neural-event-sequence-prediction/Fig3.jpg)

Effectiveness of Temporal Representation: 图3展示出了TimeMask和TimeJoint的有效性。MIMIC II数据集上面没效果，可能是加时间本来就没啥用。结论就是，用这两个东西肯定比只加时间的值到RNN里面要有效。

![Table 1 & 2](/images/time-dependent-representation-for-neural-event-sequence-prediction/Table1_2.jpg)

表1和表2也证明了加入时间的有效性。而且有些时候直接加时间可能会伤害模型的效果。

Effectiveness of Event Duration Regularization: 表1和表2证明了正则的有效性。

Learned Time Representation: 这段说的不明所以，论文里面还有错误，图画的也不清晰，没懂。

![Figure4](/images/time-dependent-representation-for-neural-event-sequence-prediction/Fig4.jpg)