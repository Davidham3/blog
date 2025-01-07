---
date: 2022-05-24 15:10:01+0000
description: 'NIPS 2021, datasets and benchmarks track, [MULTIBENCH: Multiscale Benchmarks
  for Multimodal Representation Learning](https://openreview.net/pdf?id=izzQAL8BciY)。代码：[MultiBench](https://github.com/pliang279/MultiBench)。这是个benchmark，涵盖15个数据集，10个模态，20个预测任务，6个研究领域。'
draft: false
math: true
tags:
- deep learning
- multimodal
title: 'MULTIBENCH: Multiscale Benchmarks for Multimodal Representation Learning'
---

NIPS 2021, datasets and benchmarks track, [MULTIBENCH: Multiscale Benchmarks for Multimodal Representation Learning](https://openreview.net/pdf?id=izzQAL8BciY)。代码：[MultiBench](https://github.com/pliang279/MultiBench)。这是个benchmark，涵盖15个数据集，10个模态，20个预测任务，6个研究领域。

<!--more-->

这是个benchmark，涵盖15个数据集，10个模态，20个预测任务，6个研究领域。

评估三项内容：

1.  generalization
2.  time and space complexity
3.  modality robustness

提供了20个关于融合、优化目标、训练方法的核心方法的标准实现。

# 1 Introduction

A modality refers to a way in which a signal exists or is experienced.模态是指数据的存在或表现形式。

![Figure1](/blog/images/multibench-multiscale-benchmarks-for-multimodal-representation-learning/Fig1.jpg)

**Limitations of current multimodal datasets**：现在的多模态主要研究图像和文本，其他模态太少了。此外，当前的benchmark过分关注效果，忽略了时间和空间复杂度，同时也忽略了一些不完美的模态带来的鲁棒性的下降。在实际应用中，以上三点都应该被考虑。