---
categories:
- software
date: 2018-07-14 20:35:43+0000
description: 写完公式渲染不出来，比如\vec{i}_j就会出错
draft: false
math: null
tags:
- software
title: hexo markdown mathjax 冲突问题
---
写完公式渲染不出来，比如\vec{i}_j就会出错
<!--more-->
markdown中的下划线_表示斜体，在latex中，是下标。
\\在latex中是换行，在markdown中会转义成\。
所以导致如果写公式\vec{i}_j，本来应该是向量i的j下标，就会渲染不出来。
解决方案：
修改默认的渲染器：

```
npm uninstall hexo-renderer-marked –save
```

安装 hexo-renderer-markdown-it 和 markdown-it-katex

```
npm install hexo-renderer-markdown-it --save
npm install markdown-it-katex --save
```

然后在 _config.yml 里面加入下面的代码：

```
# Markdown-it config
## Docs: https://github.com/celsomiranda/hexo-renderer-markdown-it/wiki/
markdown:
  render:
    html: true
    xhtmlOut: false
    breaks: true
    linkify: true
    typographer: true
    quotes: '“”‘’'
  plugins:
  anchors:
    level: 2
    collisionSuffix: ''

math:
  engine: 'katex'   
  katex:
    css: https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.11.1/katex.min.css
    js: https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.11.1/katex.min.js
    config:
      # KaTeX config
      throwOnError: false
      errorColor: "#cc0000"
```

然后在主题的 _config.yml 里面打开 katex 支持即可。