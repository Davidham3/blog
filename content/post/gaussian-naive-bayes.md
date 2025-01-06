---
categories:
- 机器学习
date: 2018-06-14 19:33:30+0000
description: 假设连续型随机变量服从高斯分布的朴素贝叶斯。发现自己实现的版本比sklearn的精度低了20%左右……研究了一下差在了哪里。
draft: false
math: true
tags:
- machine learning
- 已复现
title: Gaussian Naive Bayes
---
假设连续型随机变量服从高斯分布的朴素贝叶斯。发现自己实现的版本比sklearn的精度低了20%左右……研究了一下差在了哪里。
<!--more-->

# 朴素贝叶斯

朴素贝叶斯是基于贝叶斯定理与特征条件独立假设的分类器。

## 原理

朴素贝叶斯通过给定训练集

$$T = \lbrace (x\_1, y\_1), (x\_2, y\_2), ···, (x\_N, y\_N)\rbrace $$

训练学习到联合概率分布$P(X, Y)$，通过先验概率分布

$$P(Y = c\_k), k = 1,2,...,K$$

和条件概率分布

$$P(X = x \mid Y = c\_k) = P(X^{(1)} = x^{(1)}, ···, X^{(n)} = x^{(n)} \mid Y = c\_k), k=1,2,...,K$$

学习到联合概率分布$P(X, Y)$

由特征相互独立假设，可得

$$P(X = x \mid Y = c\_k) = \prod^n\_{j=1}P(X^{(j)}=x^{(j)} \mid Y = c\_k)$$

分类时，对给定的输入$x$，模型计算$P(Y = c\_k \mid X = x)$，将后验概率最大的类作为$x$的类输出，后验概率计算如下：

$$
\begin{aligned}
    P(Y = c\_k \mid X = x) &= \frac{P(X = x \mid Y = c\_k)P(Y = c\_k)}{\sum\_kP(X = x \mid Y = c\_k)P(Y = c\_k)} \\
    & = \frac{P(Y = c\_k) \prod\_j P(X^{(j)} = x^{(j)} \mid Y = c\_k)}{\sum\_k P(Y = c\_k) \prod\_j P(X^{(j)} = x^{(j)} \mid Y = c\_k)}
\end{aligned}
$$

由于分母对任意的$c\_k$都相同，故朴素贝叶斯分类器可以表示为：

$$
y = \mathop{\arg\max}\_{c\_k} P(Y = c\_k) \prod\_j P(X^{(j)} = x^{(j)} \mid Y = c\_k)
$$

## 参数估计

1. 如果特征是离散型随机变量，可以使用频率用来估计概率。

    $$P(Y = c\_k) = \frac{\sum^N\_{i=1}I(y\_i = c\_k)}{N}, k=1,2,...,K$$

    设第$j$个特征的取值的集合为${a\_{j1}, a\_{j2}, ..., a\_{js\_j}}$，则

    $$
    \begin{gathered}P(X^{(j)} = a\_{jl} \mid Y = c\_k) = \frac{\sum^N\_{i=1}I(x^{(j)}\_i = a\_{jl}, y\_i = c\_k)}{\sum^N\_{i=1}I(y\_i = c\_k)}\\
    j=1,2,...,n; \ l=1,2,...,S\_j; \ k=1,2,...,K
    \end{gathered}
    $$

2. 如果特征是连续型随机变量，可以假设正态分布来估计条件概率。

    $$P(X^{(j)} = a\_{jl} \mid Y = c\_k) = \frac{1}{\sqrt{2 \pi \sigma^2\_{c\_k,j}}}\exp{(- \frac{(a\_{jl} - \mu\_{c\_k,j})^2}{2 \sigma^2\_{c\_k,j}})}$$

    这里$\mu\_{c\_k,j}$和$\sigma^2\_{c\_k,j}$分别为$Y = c\_k$时，第$j$个特征的均值和方差。

## 代码

因为二值分类和$n$值分类是一样的，故以下代码只实现了$n$值分类的朴素贝叶斯分类器。
仓库:[https://github.com/Davidham3/naive_bayes](https://github.com/Davidham3/naive_bayes)

```python
# -*- coding:utf-8 -*-
import numpy as np
from collections import defaultdict

def readDataSet(filename, frequency = 0, training_set_ratio = 0.7, shuffle = True):
    '''
    read the dataset file, and shuffle, remove all punctuations
    
    Parameters