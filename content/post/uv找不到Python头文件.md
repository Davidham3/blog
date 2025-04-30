---
title: "uv找不到Python头文件"
description: "最近在微调LLM的时候，我发现使用uv构建的环境，有时候会找不到Python.h，导致一些库报错，如`fatal error: Python.h: No such file or directory`。通过设置`python-preference`可以解决。"
date: 2025-04-30T10:34:55Z
image: 
math: 
license: 
hidden: false
comments: true
draft: false
---

最近在微调LLM的时候，我发现使用uv构建的环境，有时候会找不到Python.h，导致一些库报错，如`fatal error: Python.h: No such file or directory`。通过设置`python-preference`可以解决。

<!--more-->

我最近使用`nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`这个镜像，在里面安装了uv。然后每个使用uv管理的项目都用这个镜像运行代码。也就是将代码下载到容器里面，通过uv sync快速拉起虚拟环境，然后通过uv run运行项目。

然而我发现在使用ms-swift进行微调的时候，就会报`fatal error: Python.h: No such file or directory`这个错误。研究后发现有几个条件同时满足的时候，就会触发这个问题：
1. 基础镜像里面安装了python（非dev版本）：我这个镜像安装的是python3.10
2. 项目使用的python版本和机器上的python版本一致：我这个项目用的是python3.10

当uv发现机器上已经安装python的时候，会默认选择使用机器上的python。详细的可以看官方文档：[Concepts: python-versions](https://docs.astral.sh/uv/concepts/python-versions/)。在uv的体系中，存在两种python，一种是managed Python，一种是system Python。前者表示uv自行安装的python，后者是系统内已经安装的python。

同时uv里面有一个设定：[python-preference](https://docs.astral.sh/uv/reference/settings/#python-preference)。这个选项默认值是`managed`。表示“相比system Python，uv会优先使用managed Python”。除此以外，还有only-managed，system，only-system这三种。

考虑到我的情况：容器里面只有1个python，是system Python，版本是3.10。我要运行的项目也是3.10。uv默认使用managed模式，因此就没有安装python，用了系统自带的python。而系统自带的这个python又是非dev版，没有Python.h这样的头文件，所以flash-attention这些需要Python.h头文件的库就会报错。

因此解决方案就是设定上文提到的：python-preference，修改为`only-managed`：
```toml
[tool.uv]
python-preference = "only-managed"
```
这个选项的含义是只使用managed Python，不用system Python。在项目的pyproject.toml里面增加上述配置后，uv sync的时候就会下载一个python3.10，并且带头文件，这样就不会报错了。