---
title: "uv加速"
description: 国内在使用uv的时候，可能会涉及到装python的速度太慢的问题，为了解决这个问题，可以使用`UV_PYTHON_INSTALL_DIR`这个环境变量。除此以外，对于多人协作场景，`UV_CACHE_DIR`也是一个有用的环境变量。本文会介绍这两个变量。
date: 2025-04-14T15:58:27Z
image: 
math: 
license: 
hidden: false
comments: true
draft: false
---

国内在使用uv的时候，可能会涉及到装python的速度太慢的问题，为了解决这个问题，可以使用`UV_PYTHON_INSTALL_DIR`这个环境变量。除此以外，对于多人协作场景，`UV_CACHE_DIR`也是一个有用的环境变量。本文会介绍这两个变量。

<!--more-->

## UV_PYTHON_INSTALL_DIR

`uv sync`、`uv venv`、`uv python install`这几个命令都会安装一个python。这个python的安装包会从[astral-sh/python-build-standalone/releases](https://github.com/astral-sh/python-build-standalone/releases)这里下载。但是对于国内的一些位置，从这里下载python的速度非常慢，有些地方根本访问不了。一个比较简单的方法是自己先进入这个页面，找到一个版本，比如`20250409`，然后下载几个需要的python版本，比如3.10、3.11、3.12，然后根据自己机器的架构，比如是x86_64的，linux系统，那就下载：
```
cpython-3.10.17+20250409-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
cpython-3.11.12+20250409-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
cpython-3.12.10+20250409-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
```
然后在本地建一个目录，比如`/mnt/workspace/uv_python_install_dir/20250409`，然后将上述压缩包放入这个目录，然后将环境变量`UV_PYTHON_INSTALL_DIR`设置成这个目录，这样就uv就会去这个目录里面找压缩包，然后快速安装python了。


## UV_CACHE_DIR

uv会把这台机器上安装过的一些源码包、wheel包存储在这个环境变量指向的目录下。因此一旦通过`uv sync`同步过一个环境，那么这个环境需要的依赖包就会在这个目录存储一份。对于云原生场景，启动一个新的容器，在这个容器里面使用`uv sync`，或者`uv pip install`的时候，如果可以将这个缓存目录挂载到容器内，然后设定环境变量`UV_CACHE_DIR`指向这个目录，那就可以快速拉起一个环境，所有的依赖都不需要重新下载了。

简单来说就是在开发机上，设定`UV_CACHE_DIR`为一个可共享的目录。然后用`uv sync`同步一个环境，此时这个目录就会存储各种缓存。然后在云原生平台启动容器的时候，挂载这个目录，并且设定容器的环境变量`UV_CACHE_DIR`为这个目录。然后在容器内使用`uv sync`，就可以利用这份缓存数据快速拉起环境。

当然，上述方法也有缺点。比如像阿里云的NAS，如果使用NFS协议挂载，由于uv在构建环境的时候是并行，存在一部分python包他们的文件是冲突的，uv的并行会让阿里云的NAS出错，会报一个OS Error 523。阿里云官方问题有讲具体原因：https://help.aliyun.com/zh/nas/user-guide/cross-mount-compatibility-faq#section-dti-749-ix0。核心问题就是在阿里云NAS上以NFS协议挂载的时候，不支持并发对一个目录的文件进行rename。这是阿里云NAS产品设计上的问题，所以使用阿里云NAS作为uv缓存的话，就需要用户自己解决了。我目前测试的结果是，像部分jupyter相关的包，会冲突，会报523。但是报了523后，可以再次执行`uv sync`，然后这个同步会继续进行，多执行几次，就可以强制安装好环境。