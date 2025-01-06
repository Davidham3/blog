---
categories:
- 论文阅读笔记
date: 2018-01-18 18:58:25+0000
description: ICDM 2008. 推荐系统：协同过滤在隐反馈数据上的应用，这个算法在GitHub上有人实现了，性能很强。这是我的阅读笔记，把论文当中的主要部分抽出来了。原文链接：[Collaborative
  Filtering for Implicit Feedback Datasets](https://ieeexplore.ieee.org/abstract/document/4781121/)
draft: false
math: true
tags:
- machine learning
- recommender system
- implicit feedback
title: Collaborative Filtering for Implicit Feedback Datasets
---
ICDM 2008. 推荐系统：协同过滤在隐反馈数据上的应用，这个算法在GitHub上有人实现了，性能很强。这是我的阅读笔记，把论文当中的主要部分抽出来了。原文链接：[Collaborative Filtering for Implicit Feedback Datasets](https://ieeexplore.ieee.org/abstract/document/4781121/)
<!--more-->
# Introduction
In this part, this paper introduce 4 important characteristics for implicit feedback:
## No negative feedback
For example, a user that did not watch a certain show might have done so because she dislikes the show or was not availale to watch it. So by observing the users behavior, we can infer which items they probably like and thus chose to consume. However, it's hard to reliably infer which item a user did not like.
## Implicit feedback is inherently noise
For example, we may view purchases behavior for an individual, but this does not necessarily indicate a positive view of thhe product. The item may have been purchased as a gift. Or a television is on a particular channel and a particular time, but the viewer might be asleep.
## The numerical value of explicit feedback indicate preference, whereas the numerical value of implicit feedback indicates confidence
Numerical values of implicit feedback describe the frequency of actions, e.g., how much time the user watched a certain show, how frequently a user is buying a certain item, etc. A larger value is not indicating a higher preference.
## Evaluation of implicit-feedback recommender requires appropriate measures
For example, if we gather data on television viewing, it's unclear how to evaluate a show that has been watched more than once, or how to compare two shows that are on at the same time, and hence cannot both be watched by the user.

# preliminaries
notions:
users $u, v$
items $i, j$
observations $r\_{ui}$, associate users and items. For explicit feedback datasets, those values would be ratings that indicate the preference by user $u$ and item $i$. For implicit datasets, $r\_{ui}$ can indicate observations for user actions. For example, $r\_{ui}$ can indicate the number of times $u$ purchased item $i$ or the time $u$ spent on webpage $i$.

# previous work
## Neighborhood models
Its original form is user-oriented, see [1] for a good analysis.
Later, an analogous item-oriented approach [2,3] became popular. In those methods, a rating is estimated using known ratings made by the same user on similar items. In addition, item-oriented methods are more amenable to explaining the reasoning behind predictions. This is because users are familiar with items previously preferred by them, but usually do not know those allegedly like minded users.
Central to most item-oriented approaches is a similarity measure between items, where $s\_{ij}$ denotes the similarity of $i$ and $j$. Frequently, it is based on the Pearson correlation coeffcient. Our goal is to predict $r\_{ui}$\--the unobserved value by user $u$ for item $i$. Using the similarity maesure, we identify the $k$ items rated by $u$, which are most similar to $i$. This set of $k$ neighbors is denoted by $S^k(i;u)$. The predicted value of $r\_{ui}$ is taken as a weighted average of the ratings for neighboring items:
$$\hat{r}\_{ui} = \frac{\sum\_{j\in S^k(i;u)}s\_{ij}r\_{uj}}{\sum\_{j\in S^k(i;u)}s\_{ij}}$$
Some enhancements of this scheme are well practiced for explicit feedback, such as correcting for biases caused by varying mean ratings of different users and items.
All item-oriented models share a disadvantage in regards to implicit feedback - they do not provide the flexibility to make a distinction between user preferences and thhe confidence we might have in those preferences.
## Latent factor models
Latent factor models comprise an alternative approach to CF with the more holistic goal to uncover latent features that explain observed ratings; example include pLSA\cite{ref4}, neural networks\cite{ref5}, and Latent Dirichlet Allocation\cite{ref6}. We will focus on models that are induced by Singular Value Decomposition(SVD) of the user-item observations matrix. Many of the recent works, applied to explicit feedback datasets, suggested modeling directly only the observed ratings, while avoiding overfitting through an adequate regularized model, such as:
$$\min \limits\_{x\_*,y\_*} \sum \limits\_{r\_{w,i}is known} (r\_{ui}-x^T\_uy\_i)^2+\lambda (\lVert x\_u\rVert^2+\lVert y\_i \rVert^2)$$
Here, $\lambda$ is used for regularizing the model. Parameters are often learnt by stochastic gradient descent;

# Our model
First, we need to formalize the notion of confidence which the $r\_{ui}$ variables measure. To this end, let us introduce a set of binary variables $p\_{ui}$, which indicates the preference of user $u$ to item $i$. The $p\_{ui}$ values are derived by binarizing the $r\_{ui}$ values:
$$p\_{ui}=
\begin{cases}
1 & r\_{ui}>0\\
0 & r\_{ui}=0
\end{cases}$$
In other words, if a user $u$ consumed item $i$($r\_{ui}>0$), then we have an indication that $u$ likes $i$($p\_{ui}=1$). On the other hand, if $u$ never comsumed $i$, we believe no preference($p\_{ui}=0$).
We will have different confidence levels also among items that are indicated to be preferred by the user. In general, as $r\_{ui}$ grows, we have a stronger indication that the user indeed like thhe item. Consequently, we introduce a set of variables, $c\_{ui}$, which measure our confidence in observing $p\_{ui}$. A plausible choice for $c\_{ui}$ would be:
$$c\_{ui} = 1 + \alpha r\_{ui}$$
This way, we have some minimal confidence in $p\_{ui}$ for every user-item pair, but as we observe more evidence for positive preference, our confidence in $p\_{ui}=1$ increases accordingly. The rate of increase is controlled by the constant $\alpha$. In our experiments, setting $\alpha = 40$ was found to produce good results.
Our goal is to find a vector $x\_u\in \mathbb{R}^f$ for each user $u$, and a vector $y\_i\in \mathbb{R}^f$ for each item $i$ that will factor user preferences. These vectors will be known as the user-factors and the item-factors, respectively. Preferences are assumed to be the inner products: $p\_{ui}=x^T\_uy\_i$. Essentially, the vectors strive to map users and items into a common latent vector space where they can be directly compared. This is similar to matrix factorization techniques which are popular for explicit feedback data, with two important distinction: (1) We need to account for the varying confidence levels, (2) Optimization should account for all possible $u, i$ pairs, rather than only these corresponding to observed data. Accordingly, factors are computed by minimizing the following cost function:
$$\min \limits\_{x\_*, y\_*}\sum \limits\_{u,i}c\_{ui}(p\_{ui}-x^T\_uy\_i)^2+\lambda(\sum\limits\_{u}\lVert x\_u\rVert^2+\sum\limits\_{i}\lVert y\_i\rVert^2)$$
The $\lambda(\sum\limits\_{u}\lVert x\_u\rVert^2+\sum\limits\_{i}\lVert y\_i\rVert^2)$ term is necessary for regularizing the model such that it will not overfit the training data.


[1]. Herlocker J L, Konstan J A, Borchers A, et al. An algorithmic framework for performing collaborative filtering[C]. international acm sigir conference on research and development in information retrieval, 1999: 230-237.
[2]. Linden G, Smith B, York J C, et al. Amazon.com recommendations: item-to-item collaborative filtering[J]. IEEE Internet Computing, 2003, 7(1): 76-80.
[3]. Sarwar B M, Karypis G, Konstan J A, et al. Item-based collaborative filtering recommendation algorithms[J]. international world wide web conferences, 2001: 285-295.
[4]. Hofmann T. Latent semantic models for collaborative filtering[J]. ACM Transactions on Information Systems, 2004, 22(1): 89-115.
[5]. Salakhutdinov R, Mnih A, Hinton G E, et al. Restricted Boltzmann machines for collaborative filtering[C]. international conference on machine learning, 2007: 791-798.
[6]. Blei D M, Ng A Y, Jordan M I, et al. Latent Dirichlet Allocation[C]. neural information processing systems, 2002, 3(0): 601-608.