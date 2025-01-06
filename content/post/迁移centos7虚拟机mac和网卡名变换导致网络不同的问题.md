---
categories:
- 分布式平台
date: 2018-06-12 11:15:17+0000
description: vcenter迁移虚拟机的时候，迁移之后虚拟机网络不通。
draft: false
math: null
tags:
- virtual machine
title: 迁移CentOS7虚拟机mac和网卡名变换导致网络不通的问题
---
vcenter迁移虚拟机的时候，迁移之后虚拟机网络不通。
<!--more-->

参考：[解决CentOS 7虚拟机克隆的网络问题](https://www.jianshu.com/p/29af2068cfb6)

使用vcenter的迁移后，虚拟机出现了网络不通的现象，仔细观察可以发现vcenter给虚拟机分配了新的mac地址。
因为Linux系统会记录mac地址与网卡名的关系，所以Linux系统在运行后，发现mac变了，于是会给当前这张网卡分配一个新的网卡名。
解决方案就是：
1. 修改网卡配置文件/etc/sysconfig/network-scripts/ifcfg-eno16884287
	删除UUID这一行，因为每张网卡的mac地址是不一样的，所以UUID也是不一样的。
	修改HWADDR为虚拟机克隆后的MAC地址
2. 进入/etc/udev/rules.d/这个目录，将里面的.rules文件改名
	`mv 70-persistent-ipoib.rules 70-persistent-ipoib.rules.bak`
	`mv 90-eno-fix.rules 90-eno-fix.rules.bak`
3. 重启
	`reboot`