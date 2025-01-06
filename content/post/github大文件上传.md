---
categories:
- software
date: 2019-12-20 11:18:36+0000
description: 刚才开源了我们组在AAAI 2020上一篇论文的代码和数据，上传数据的时候超了GitHub的100M大小限制，GitHub说让我用lfs解决，研究了一下怎么传，记录一下，以后说不定还会用到。
draft: false
math: null
tags:
- software
title: github大文件上传
---
刚才开源了我们组在AAAI 2020上一篇论文的代码和数据，上传数据的时候超了GitHub的100M大小限制，GitHub说让我用lfs解决，研究了一下怎么传，记录一下，以后说不定还会用到。
<!--more-->

```
git lfs install

git lfs track "data.tar.gz"

git add .gitattributes

git commit -m "Updated attributes"

git push

git add data.tar.gz

git lfs ls-files

git commit -m "Add file"

git push
```

原理还不太懂，这几天太忙了，过几天看看。

2020.1.2 更新一下：

最近收到了GitHub的邮件，说这个LFS是有限额的。。。然后我的已经超额了。。。

这个东西是分storage和bandwidth两个额度，两个额度都是1个月1G。存储的话，如果你存储一次，就会占一次的容量，如果你修改了里面的内容，再push一次，就会再占一次这么大的空间，比如有一个100M的文件，我push上去，占100M，我改了一下，再push，那我就使用了200M空间。。。这个空间是这么计算的。。。

带宽的计算是，你上传文件不算流量。但是只要有人下载，那就走这个流量，一个月1G免费流量。。。我这个数据集，一天就超了。。。

解决方法是什么呢？进入GitHub的help page里面，可以找到删除文件的方法。

但是，即便你想办法删了，如果你的仓库之前被别人fork过了，那他们那边的下载，也会占用你的bandwidth，得知这点后我真的好无语。。。主要另一个坑爹的事情是，这个storage和bandwidth是按账户记得，不是按仓库，即一个账户1月1G。。。

最后我把仓库删了，想着去联系那几个fork过的人，让他们删除他们的仓库，但是他们竟然没有留下邮箱。。。所以我根本联系不上他们。。。简直了。。。