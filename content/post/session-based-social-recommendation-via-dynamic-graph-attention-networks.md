---
categories:
- 论文阅读笔记
date: 2019-05-29 13:37:55+0000
description: WSDM 2019，原文链接：[Session-based Social Recommendation via Dynamic Graph
  Attention Networks](https://arxiv.org/abs/1902.09362)
draft: false
math: true
tags:
- deep learning
- Attention
- Graph
title: Session-based Social Recommendation via Dynamic Graph Attention Networks
---
WSDM 2019，原文链接：[Session-based Social Recommendation via Dynamic Graph Attention Networks](https://arxiv.org/abs/1902.09362)

<!--more-->

# Abstract

像 Facebook 和 Twitter 这样的在线社区很流行，已经成为很多用户生活中的重要部分。通过这些平台，用户可以发掘并创建信息，其他人会消费这些信息。在这种环境下，给用户推荐相关信息变得很重要。然而，在线社区的推荐是一个难题：1. 用户兴趣是动态的，2. 用户会受其朋友的影响。此外，影响者与环境相依。不同的朋友可能关注不同的话题。对两者建模对推荐来说是重要的。

我们提出了一个基于动态图注意力机制的在线社区推荐系统。我们用一个 RNN 对动态的用户行为建模，用图卷积对依赖环境的社交影响建模，可以动态地根据用户当前的兴趣推测影响者。整个模型可以高效地用于大规模的数据。几个真实数据集上的实验结果显示我们的方法很好，源码在：[https://github.com/DeepGraphLearning/RecommenderSystems](https://github.com/DeepGraphLearning/RecommenderSystems)

# 1 Introduction

在线社区已经成为今天在线体验的重要组成部分。Facebook, Twitter, 豆瓣可以让用户创建、分享、消费信息。因此这些平台的推荐系统对平台上的表层信息和维持用户活跃度来说很重要。然而，在线社区对推荐系统提出了一些挑战。

首先，用户兴趣本质上来说是动态的。一个用户可能一段时间对体育感兴趣，之后呢对音乐感兴趣。其次，因为在线社区里面的用户经常给朋友分享信息，用户也会被他们的朋友影响。举个例子，一个找电影的用户可能会被她的朋友喜欢的电影影响。此外，施加影响的一方组成的集合是动态的，因为这和环境有关。举个例子，一个用户在找一个搞笑电影的时候会听取一群喜欢喜剧的朋友的意见，在找动作电影的时候，会受到另一组朋友的影响。

**Motivating Example.** 图 1 展示了 Alice 和 她的朋友在一个在线社区的行为。行为通过一个动作（比如点击操作）序列描述。为了捕获用户的动态兴趣，她们的行为被分成了不同的子序列，表示 *sessions*。我们感兴趣的是基于 *session* 的推荐：我们根据当前情境下 Alice 已经消费过的东西给她推荐下一个她可能消费的东西。图 1 展示出两个情景，a 和 b。此外，Alice 朋友们的消费信息也是可获得的。我们会利用这些信息生成更好的推荐。因此我们在一个基于 session 的社交推荐情景下。

![Figure1](/images/session-based-social-recommendation-via-dynamic-graph-attention-networks/Fig1.JPG)

在 session a 中，Alice 浏览了体育的物品。她的两个朋友：Bob 和 Eva，是出了名的体育粉丝（长期兴趣），他们最近正好浏览了体育相关的物品（短期兴趣）。考虑到这个情况，Alice 可能被他们两个影响，比如说接下来她可能会学习乒乓球。在 session b 中，Alice 对文学艺术物品感兴趣。这个环境和刚才不一样了因为她没有最近正在消费这样物品的朋友。但是 David 一直对这个话题感兴趣（长期兴趣）。在这种情况下，对 Alice 来说可能会被 David 影响，可能会被推荐一本 David 喜欢的书。这些例子表明了一个用户当前的兴趣是如何与他不同的朋友的兴趣相融合来提供基于情景的推荐的。我们提出了一个推荐模型来处理这两种情况。

当前的推荐系统要么对用户的动态兴趣建模，要么对他们的社交影响建模，但是，据我们所知，现存的方法还没有融合过他们。最近的一个研究对 session 级别的用户行为使用 RNN 建模，忽略了社交影响。其他的研究仅考虑社交影响。举个例子，Ma et al. 探索了朋友的长期兴趣产生的社交影响。但是，不同用户的影响是静态的，没有描绘出每个用户当前的兴趣。

我们提出了一个方法对用户基于 session 的兴趣和动态社交影响同时建模。也就是说，考虑了基于当前用户的 session，他的朋友的哪个子集影响了他。我们的推荐模型基于动态注意力网络。我们的方法先用一个 RNN 对一个 session 内的用户行为建模。根据用户当前兴趣——通过 RNN 的隐藏表示捕获到的——我们使用 GAT 捕获了他的朋友的影响。为了提供 session 级别的推荐，我们区分了短期兴趣和长期兴趣。在给定用户当前兴趣的基础上，每个朋友的影响通过注意力机制自动地决定。

我们做了大量实验，效果比很多方法好。贡献如下：
- 提出了同时对动态用户兴趣和依赖环境的社交影响学习后对在线社区进行推荐的方法。
- 提出了基于动态图注意力网络的推荐方法。在大数据集上也有效。
- 实验结果比 state-of-the-art 好很多。

# 2 Related Work

讨论三条路线：1. 对动态用户行为建模的推荐系统，2. 考虑社交影响的推荐系统，3. 图卷积网络。

## 2.1 Dynamic Recommendation

## 2.2 Social Recommendation

## 2.3 Graph Convolutional Networks

# 3 Problem Definition

推荐系统根据历史行为推荐相关的物品。传统的推荐模型，如矩阵分解，忽略了用户的消费顺序。在线社区中，用户兴趣是快速变化的，必须要考虑用户偏好顺序，以便对用户的动态兴趣建模。实际上，因为用户全部的历史记录可以很长（有些社区存在好多年了），用户兴趣切换的很快，一个常用的方法是将用户的偏好分成不同的 session，（使用时间戳，以一个星期为时间段考虑每个用户的行为）并以 session 为级别提供推荐。定义如下：

DEFINITION 1. (**Session-based Recommendation**)，$U$ 表示用户的集合，$I$ 表示物品集。每个用户 $u$ 和一组带有时间步 $T$ 的 session 相关，$I^u\_T = \lbrace \vec{S}^u\_1, \vec{S}^u\_2, \dots \vec{S}^u\_T \rbrace$，其中 $\vec{S}^u\_t$ 是用户 $u$ 的第 $t$ 个 session。在每个 session 内，$\vec{S}^u\_t$ 由一个用户行为的序列 $\lbrace i^u\_{t,1}, i^u\_{t,2}, \dots, i^u\_{t,N\_{u,t}} \rbrace$ 组成，其中 $i^u\_{t,p}$ 是在第 $t$ 个 session 中用户消费的第 $p$ 个物品，$N\_{u,t}$ 是 session 中物品的总数。对于每个用户 $u$，给定一个 session $\vec{S}^u\_{T+1} = \lbrace i^u\_{T+1,1}, \dots i^u\_{T+1,n} \rbrace$，基于 session 的推荐系统的目标是从 $I$ 中推荐一组用户可能在下来的 $n+1$ 步时感兴趣的物品，即 $i^u\_{T+1, n+1}$。

在在线社区中，用户的兴趣不仅与他们的历史行为相关，也受他们的朋友的影响。举个例子，一个朋友看电影，我也可能会感兴趣。这就叫社交影响。此外，从朋友那里来的影响是跟环境有关的。换句话说，从朋友那里来的影响是不一样的。如果一个用户想买个笔记本电脑，她可能更倾向于问问她喜欢高科技产品的朋友；如果她要买相机，她可能会被她的摄影师朋友影响。就像图 1，一个用户可能被她朋友的长期兴趣和短期兴趣影响。

为了提供一个有效的推荐结果，我们提出对动态的用户兴趣和依赖于环境的社交影响建模。我们定义了如下的问题：

DEFINITION 2. (**Session-based Social Recommendation**) $U$ 表示用户集，$I$ 表示物品集合，$G=(U, E)$ 是社交网络，$E$ 是社交网络的边。给定用户 $u$ 的一个 session $\vec{S}^u\_{T+1} = \lbrace i^u\_{T+1,1}, \dots i^u\_{T+1,n} \rbrace$，目标是利用她的动态兴趣（$\cup^{T+1}\_{t=1} \vec{S}^u\_t$）和社交影响（$\cup^{N(u)}\_{k=1} \cup^T\_{t=1} \vec{S}^k\_t$，其中 $N(u)$ 是用户 $u$ 的邻居），从 $I$ 中推荐一组用户 $u$ 可能在下来的 $n+1$ 步时感兴趣的物品，即 $i^u\_{T+1, n+1}$。

# 4 Dynamic Social Recommender Systems

我们提出的模型 Dynamic Graph Recommendation (DGREC) 是个动态图注意力模型，可以对用户近期的偏好和他的朋友的偏好建模。

DGREC 有 4 个模块（图 2）。首先，一个 RNN 对用户当前 session 中的物品序列建模。她朋友的偏好使用长期偏好和短期偏好融合来建模。短期偏好，或是最近一次 session 中的物品也使用 RNN 来编码。朋友的长期偏好通过一个独立的嵌入层编码。模型使用 GAT 融合当前用户的表示和她朋友的表示。这是我们模型的关键：我们提出了基于用户当前的兴趣学习每个朋友的权重的机制。最后一步，模型通过融合用户当前偏好和她的社交影响得到推荐结果。