---
date: 2025-01-06 12:41:49+0000
description: 记录一下关于TrajDL的信息。TrajDL是我编写的第一个体系完整的python工具包，面向轨迹深度学习，目的是支持很多轨迹深度学习算法，包括轨迹表示学习、轨迹分类、下一位置预测等方法。目前已经在GitHub上开源，项目主页是：[https://github.com/Spatial-Temporal-Data-Mining/TrajDL](https://github.com/Spatial-Temporal-Data-Mining/TrajDL)，已经发布了0.1.0版本，可以在pypi上下载，官方文档是：[https://trajdl.readthedocs.io/en/latest/](https://trajdl.readthedocs.io/en/latest/)，可以通过这个文档查阅它的使用方法。
draft: false
math: null
tags:
- deep learning
- trajectory
title: Trajdl
---

记录一下关于TrajDL的信息。TrajDL是我编写的第一个体系完整的python工具包，面向轨迹深度学习，目的是支持很多轨迹深度学习算法，包括轨迹表示学习、轨迹分类、下一位置预测等方法。

目前已经在GitHub上开源，项目主页是：[https://github.com/Spatial-Temporal-Data-Mining/TrajDL](https://github.com/Spatial-Temporal-Data-Mining/TrajDL)，已经发布了0.1.0版本，可以在pypi上下载，官方文档是：[https://trajdl.readthedocs.io/en/latest/](https://trajdl.readthedocs.io/en/latest/)，可以通过这个文档查阅它的使用方法。

<!--more-->

目前0.1.0版本只支持了TULER、t2vec、GMVSAE、ST-LSTM 4个算法，当然还开发了比如CTLE、HIER这样的算法，但是还没有开发完。

TrajDL的核心优势在于帮用户管理了数据集，用户可以快速下载公开数据集，并且通过TrajDL设计的dataset高效地完成实验，一些关键算子通过C++实现，数据集基于Arrow构建，所以整个框架在性能、内存使用上都有比较好的效果。整个训练过程构建在lightning上，所以即支持API开发训练代码，又支持通过配置文件进行训练。

0.2.0版本主要会在轨迹相似度计算上面实现一些算法，目前还在开发中。