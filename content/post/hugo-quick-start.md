---
date: 2024-12-24 06:00:29+0000
description: 本文讲一下如何快速启动一个Hugo博客。我之前的博客是部署在hexo上的，而且已经很久用写过了，现在要重启，由于hexo的环境部署依赖node，想换一套更好用的系统，Claude推荐我使用Hugo。
draft: false
math: null
tags:
- blog
title: Hugo Quick Start
---

本文讲一下如何快速启动一个Hugo博客。我之前的博客是部署在hexo上的，而且已经很久用写过了，现在要重启，由于hexo的环境部署依赖node，想换一套更好用的系统，Claude推荐我使用Hugo。

<!--more-->

Hugo现在出到0.140.1版本了，因此本文是按照这个版本编写的。

先构建一个docker镜像：

```dockerfile
# 使用官方 Go 镜像作为基础镜像
FROM golang:latest

# 安装 Git
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装 Dart Sass
RUN curl -fsSL https://github.com/sass/dart-sass/releases/download/1.83.0/dart-sass-1.83.0-linux-x64.tar.gz \
    | tar -xz -C /usr/local/bin/ --strip-components=1 \
    && chmod +x /usr/local/bin/sass

# 安装 Hugo
RUN curl -L https://github.com/gohugoio/hugo/releases/download/v0.140.1/hugo_0.140.1_linux-amd64.tar.gz \
    | tar -xz -C /usr/local/bin/ hugo \
    && chmod +x /usr/local/bin/hugo

# 设置工作目录
WORKDIR /app

# 验证安装
RUN go version && \
    git --version && \
    sass --version && \
    hugo version

# 设置容器启动时的默认命令
CMD ["bash"]
```

通过下面的命令构建镜像：

```bash
docker build -t hugo:0.140.1 .
```

构建完成后通过下面的命令启动：

```bash
docker run -ti --rm --user $(id -u):$(id -g) -v $PWD:/app -p 1314:1314 hugo:0.140.1 bash
```

创建一个新的站点：

```bash
# 创建一个新的站点叫blog，配置文件使用yaml（否则会用toml）
hugo new site blog --format=yaml

cd blog

git init

# 这里安装hextra这个主题
git submodule add https://github.com/imfing/hextra.git themes/hextra
```

然后去blog/hugo.yaml里面新增`theme: hextra`

```bash
hugo server --bind 0.0.0.0 --port 1314
```

通过上述命令启动server，然后就可以直接打开浏览器访问`localhost:1314`。

另外，我编写了一个python脚本用来将hexo的博客迁移到hugo上面，命名位`scripts/migrate.py`，直接使用`uv run scripts/migrate.py`运行即可。