---
categories:
- software
date: 2019-04-19 15:02:46+0000
description: ubuntu /boot 满了怎么办，[解决方案](https://askubuntu.com/questions/171209/my-boot-partition-hit-100-and-now-i-cant-upgrade-cant-remove-old-kernels-to)
draft: false
math: null
tags:
- software
title: ubuntu /boot 满了怎么办
---
ubuntu /boot 满了怎么办，[解决方案](https://askubuntu.com/questions/171209/my-boot-partition-hit-100-and-now-i-cant-upgrade-cant-remove-old-kernels-to)

<!--more-->

```
uname -a
```

看一下现在用的是什么内核

```
cd /boot
```

把老的内核挪走，挪的时候按版本号挪，从老的开始挪，具有同一个版本号的文件同时挪走，挪几个老的就行。

然后

```
apt-get install -f
```

```
dpkg --get-selections |grep linux-
```

把老的内核都卸载掉：

```
apt-get purge linux-headers-4.4.0-137-generic linux-image-4.4.0-137-generic linux-image-extra-4.4.0-137-generic
```

然后把刚才挪走的文件再挪回去，再用上面的命令卸载掉。