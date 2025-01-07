---
date: 2022-04-26 14:38:05+0000
description: 这是一篇讲事件序列度量学习的文章，提出的模型叫MeLES，Metric Learning for Event Sequences。[Event
  sequence metric learning](https://arxiv.org/abs/2002.08232)
draft: false
math: true
tags:
- deep learning
- event sequence
title: Event sequence metric learning
---

这是一篇讲事件序列度量学习的文章，提出的模型叫MeLES，Metric Learning for Event Sequences。[Event sequence metric learning](https://arxiv.org/abs/2002.08232)

<!--more-->

事件序列主要是指连续时间下的离散事件。比如用户的信用卡交易、在网站上的点击行为等等。

# 1. 模型架构

![Figure1](/blog/images/event-sequence-metric-learning/Fig1.jpg)

# 2. 原理

## 2.1 样本构成

每 $N$ 个序列构成一个batch，对这个batch内的序列进行切分，有三种切分方式：

1.  保持事件顺序，随机不放回采样
2.  随机切分序列，子序列之间不重叠
3.  随机切分序列，子序列之间重叠

不管怎么切，都能把一个序列切成多个子序列，这里将每个序列切成 $K$ 个子序列，那么一个batch就可以得到 $N \times K$ 个子序列。

在这些子序列中，来自同一个序列的两个子序列组成的pair，为正样本，来自不同序列的两个子序列组成的pair为负样本。这样，对于每个子序列，就有 $K - 1$ 个正样本，$(N - 1) \times K$ 个负样本。

## 2.2 序列表示

然后使用RNN或者是transformer对子序列进行编码，编码后可以得到一个向量，这个向量就是这个子序列的嵌入表示。拿到这个表示，就可以计算损失了。

## 2.3 损失

计算损失的时候，最简单的想法肯定就是类似交叉熵一样的损失，正样本的损失加上负样本的损失即可。

但是之前的研究认为，有些嵌入表示，他们的距离过于远，这种样本对模型训练没什么用，因此本文给了两个损失函数来剔除这种情况。一个叫contrastive loss，一个叫margin loss，原理都是一样的。

![loss](/blog/images/event-sequence-metric-learning/loss.jpg)

对于contrastive loss来说，正样本的损失正常计算，而负样本的损失，如果pair中的两个表示的距离大于 $m$ ，就不要了。

对于margin loss是同样的原理，$b + m$ 和 $b - m$ 构成了损失函数的边界，正样本的距离要大于 $b - m$ 才有意义，而负样本的距离要小于 $b + m$ 才有意义。而且，当 $b < m$ 的时候，这个损失就会变得和上面的contrastive loss一样，只考虑负样本的margin，因为只要距离是欧式距离，在任何情况下 $max(0, D^i\_W - b + m) = D^i\_W - b + m > 0$。

## 2.4 负样本采样策略

然后，除了上述的损失函数可以控制两个距离过远的负样本不计算损失，还可以做负样本采样，也就是刚才说的 $(N - 1) \times K$ 个负样本，只取出一部分用来训练。这里有4种方式：

1.  随机采样
2.  难例挖掘，对每个整理生成$k$个难例作为负样本
3.  负样本采样的时候考虑距离因素
4.  第四个没看明白他在说啥，倒是给了个参考文献：Florian Schroff, Dmitry Kalenichenko, and James Philbin. 2015. FaceNet: A  
    unified embedding for face recognition and clustering. 2015 IEEE Conference on  
    Computer Vision and Pattern Recognition (CVPR) (2015), 815–823.

不管是上述哪种负采样方案，除了第一种，都是要算距离的，也就是算两个embedding之间的距离。而且是要算batch内任意两个embedding之间的距离，或者说是算 $(N - 1) \times K$ 个距离。如果用欧氏距离计算嵌入 $A$ 和 $B$ 之间的距离，那么 $D(A, B) = \sqrt{\sum\_i (A\_i - B\_i)^2} = \sqrt{\sum\_i A^2\_i + \sum\_i B^2\_i - 2 \sum\_i A\_i B\_i}$，这里为了计算简便，只要让 $\Vert A \Vert = \Vert B \Vert = 1$ 就好了，那就能转换成 $D(A, B) = \sqrt{2 - 2(A \cdot B)}$。所以，为了达成上面的目标，让 $A$ 和 $B$ 的模等于1，只要对这些嵌入表示做标准化，就可以实现了。论文里面说，做了这个操作之后，负样本采样的计算复杂度是 $O(n^2h)$，这个我还没想明白，后面再说吧。

# 3. 实验

两个数据集都是银行交易数据，主要是通过交易事件序列预测用户的年龄与性别。

## 3.1 Baselines

1.  手工特征+GBM，手工构建了近1k个特征，然后用LightGBM。
    
2.  Contrastive Predictive Coding(CPC)，一个自监督学习方法，Aäron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation  
    Learning with Contrastive Predictive Coding. CoRR abs/1807.03748 (2018).  
    arXiv:1807.03748 [http://arxiv.org/abs/1807.03748](http://arxiv.org/abs/1807.03748)
    
3.  除了上面两个方法，作者还试了编码器网络+分类网络直接用于监督学习任务，这里就没有预训练了。
    

## 3.2 参数选择

![result](/blog/images/event-sequence-metric-learning/Table4_to_7.jpg)

上面4个表的结论：

1.  不同编码器效果不同
2.  在训练集上表现最好的损失函数在测试集上不一定是最好的
3.  随机slice比随机采样更好
4.  难例挖掘带来的提升是显著的（但是论文前边根本没仔细介绍难例挖掘好吧。。。）

![Figure2](/blog/images/event-sequence-metric-learning/Fig2.jpg)

图2是说嵌入在800维的时候效果最好，用bias-variance来解释。维数少的时候高bias，信息丢失，维数高的时候高variance，噪声多了。

![Figure3](/blog/images/event-sequence-metric-learning/Fig3.jpg)

图3一样，256到2048比较平缓，下游任务的效果没有明显增强。

作者说嵌入维数的增加，训练时间和显存消耗都是线性增加的。

## 3.3 嵌入可视化

tSNE，染色是用数据集中的target value染色的。学习完全是自监督的。交易序列表示的是用户的行为，因此模型可以捕获行为模式，产出的embedding如果相近，则说明用户的行为模式相似。

![Figure4](/blog/images/event-sequence-metric-learning/Fig4.jpg)

## 3.4 结果

![Table8](/blog/images/event-sequence-metric-learning/Table8.jpg)

对比手工构建的特征，模型效果强劲。fine-tuned的表示效果最好。另外可以看到的是，使用手工特征+事件序列嵌入表示的模型效果比纯手工特征效果更好。

### 3.4.1 关于半监督的实验

只取了一部分标签做实验，就像监督学习一样用手工特征的lightgbm和CPC。对于嵌入生成方法（MeLES和CPC），分别使用lightgbm和fine-tuned模型来评估效果。同时还比了监督模型在这些label上的效果。

![Figure5](/blog/images/event-sequence-metric-learning/Fig5.jpg)

![Figure6](/blog/images/event-sequence-metric-learning/Fig6.jpg)

![Figure7_and_Figure8](/blog/images/event-sequence-metric-learning/Fig78.jpg)

结论就是标签少的时候，效果很好。

# 4. 结论

提出了MeLES，效果很好，而且还可以在半监督中做预训练。好处是基本不用怎么对数据做处理就可以拿到嵌入表示，获得好的效果。而且在新的事件加入的时候，甚至是可以增量更新已经计算的嵌入表示。另一方面是嵌入表示无法还原原始的事件序列，可以起到数据加密的作用。

这里提到的增量更新其实就是，RNN的计算只要上一个时间步的信息就好了，不需要从头再训练一次，因此如果有新的事件到来，从最后一次的状态开始算就好了，这就叫增量更新。