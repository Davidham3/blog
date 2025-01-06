---
date: 2024-12-24 07:17:09+0000
draft: false
math: true
tags:
- Trajectory
title: NEUTRAJ
---

[ICDE 2019：Computing Trajectory Similarity in Linear Time: A Generic Seed-Guided Neural Metric Learning Approach](https://ieeexplore.ieee.org/abstract/document/8731427).

提出的方法叫NEUTRAJ，主要解决轨迹相似度计算时的性能问题，在线性时间内计算完毕。这个方法是要用户提前选择一种非学习的度量方式，然后这个框架可以去拟合这个度量方式的值，因此这个框架可以支持很多度量方式，比如DTW，Frechet等。模型层面有两个创新，在RNN上增加了带有记忆模块的attention机制；另外是在损失的时候增加了排序信息。

<!--more-->

# Definition

文章里面的轨迹使用$T$表示，$f(T_i, T_j)$ 表示相似度，这个相似度可以使用DTW，Frechet，可以使用各种非学习的度量方式。但是很多这样的度量方法的时间复杂度都达到了平方级，本文的一个目标就是学习一个 $g(\cdot, \cdot)$，这个函数的时间复杂度要达到 $O(n)$，训练方式就是将 $\vert f(T_i, T_j) - g(T_i, T_j) \vert$ 最小化。

# Overview

$\mathfrak{S}$ 是从所有轨迹里面随机取出的 $N$ 条轨迹，这批轨迹称为种子。然后通过 $f$ 计算一个矩阵 $\mathbf{D} \in \mathbb{R}^{N \times N}$ 出来，然后做一个normalization，得到 $\mathbf{S}$。对于任意两条轨迹，NEUTRAJ会生成两个 $d$ 维向量 $\mathbf{E_i}, \mathbf{E_j}$，使得 $f(T_i, T_j) \approx g(T_i, T_j)$。

## 预处理

将空间划分成网格，将轨迹序列转换成位置序列，位置就是网格的id。假设网格的大小是 $P \times Q$，那么定义一个memory tensor $\mathbf{M} \in \mathbb{R}^{P \times Q \times d}$，训练前全都置为0。这个tensor会和RNN一起训练。

## 带有记忆模块的RNN

$$
(\mathbf{f_t, i_t, s_t, o_t}) = \sigma (\mathbf{W_g} \cdot X^c_t + \mathbf{U_g \cdot h_{t-1} + b_g})
$$

$$
\mathbf{\tilde{c_t}} = \tanh (\mathbf{W_c} \cdot X^c_t + \mathbf{U_c \cdot h_{t-1} + b_c})
$$

$$
\mathbf{\hat{c_t} = f_t \cdot c_{t-1} + i_t \cdot \tilde{c_t}}
$$

$$
\mathbf{c_t} = \mathbf{\hat{c_t} + s_t} \cdot \text{read}(\mathbf{\hat{c_t}}, X^g_t, \mathbf{M})
$$

$$
\text{write}(\mathbf{c_t, s_t}, X^g_t, \mathbf{M})
$$

$$
\mathbf{h_t} = \mathbf{o_t} \cdot \tanh(\mathbf{c_t})
$$

where $\mathbf{W_g} \in \mathbb{R}^{4d \times 2}, \mathbf{U_g} \in \mathbb{R}^{4d \times d}, \mathbf{W_c} \in \mathbb{R}^{d \times 2}, \mathbf{U_c} \in \mathbb{R}^{d \times d}$。

这里需要注意的是 $X^c_t, X^g_t$ 分别表示轨迹的经纬度点和经纬度点所在的网格。读写操作就是对 $\mathbf{M}$ 进行读取和写入。

读操作使用两个输入，一个是网格输入 $X^g_t$，另一个是中间细胞状态 $\mathbf{\hat{c_t}}$，然后会输出一个向量 $\mathbf{c^{his}_t}$，这个是用来增强 $\mathbf{\hat{c_t}}$ 的。大体原理就是通过当前的这个网格位置，以它为矩形中心，去读取周围一个正方形区域内这些网格的memory，然后通过一个注意力机制将其变成一个 $d$ 维向量。假设正方形区域的边长是5个格子的长度，那么就会扫描一个 $5 \times 5$ 的区域，获得这些memory。

注意力机制的计算公式：

$$
\mathbf{A} = \text{softmax}(\mathbf{G_t \cdot \hat{c_t}})
$$

$$
\mathbf{mix = G^T_t \cdot A}
$$

$$
\mathbf{c^{cat}_t = [\hat{c_t}, mix]}
$$

$$
\mathbf{c_{t}^{his}} = \tanh ( \mathbf{W_{his} \cdot c_{t}^{his} + b_{his}})
$$

这里的 $G_t \in \mathbb{R}^{(2w + 1)^2 \times d}$。

写入操作：

$$
\mathbf{M}(X_g)_{new} = \sigma(\mathbf{s_t}) \cdot \mathbf{c_t} + (1 - \sigma(\mathbf{s_t})) \cdot \mathbf{M}(X_g)\_{old}
$$

## Metric Learning Procedures

RNN最后的隐藏状态作为轨迹表示。

作者说直接去拟合 $\mathbf{S}$ 会过拟合，所以要给MSE加权。具体做法是从种子里面选出一个anchor轨迹，然后看这条轨迹在 $\mathbf{S}$ 里面的哪一行，取出这一行 $\mathbf{I_a}$，然后进行采样。首先用这个向量的权重采样正样本 $n$ 个，然后用 $1 - \mathbf{I_a}$ 采 $n$ 个负样本。然后给他们按相似和不相似分别排好序，给他们赋予权重，权重是 $\mathbf{r} = (1, 1/2, \dots, 1/l, \dots, 1/n)$，然后再除以 $\sum^n_{l=1} r_l$ 进行归一化。

然后定义正样本损失函数为

$$
L^s_a = \sum^n_{l=1} r_l \cdot (g(T_a, T^s_l)- f(T_a, T^s_l))^2
$$

负样本损失：

$$
L^d_a = \sum^n_{l=1} r_l \cdot [\text{ReLU}(g(T_a, T^d_l) - f(T_a, T^d_l))]^2
$$

我觉得不相似样本的损失函数设计的不对，如果 $f$ 是DTW这类算法，DTW值越小说明两条轨迹越相似。那么如果 $g > f$，那就说明模型认为两条轨迹的距离比真实距离大。当前这些样本本身就是不相似的，所以 $f$ 应该会非常大，而如果 $g > f$ ，说明模型认为他们更不相似，此时是合理的，如果 $g < f$，说明模型认为这些样本是相似的，此时应该优化。因此如果 $ g < f$，即 $ g - f < 0$，那么此时应该让模型去优化参数，如果 $g - f > 0$，就不用优化了，因此这里的损失函数里面缺少了一个负号。通过阅读源码，发现作者使用是 $ f - g $，因此这里可以认为是论文撰写错误。

然后把两个损失相加，得到最终的loss

$$
L_{\mathfrak{S}} = \sum_{a \in [1, \dots, N]} (L^s_a + L^d_a)
$$

# Experiments

实验是在Geolife和Porto上面做的。做了一个top-k搜索，还做了一个聚类。