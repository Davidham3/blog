---
categories:
- 分布式平台
date: 2017-08-21 16:13:25+0000
draft: false
math: null
tags:
- Hadoop
- software
title: Hadoop HA安装一：在Linux中安装和配置ntp，ssh和jdk
---
Hadoop HA安装一：安装和配置ntp，ssh和jdk
<!--more-->

记录一下安装5节点的高可用Hadoop集群的安装过程。High availability表示高可用，在Hadoop集群中，表示两个主节点。HDFS中是两个Namenode，Yarn中是两个ResourceManager。本教程中的Hadoop和HBase均为HA，MySQL为双机热备。

操作系统：CentOS 7.

软件服务如下：

|软件|版本|路径|
|- | - | - |
|JDK|1.7.80|/usr/local/jdk1.7.0_80|
|MySQL|5.6.37|/usr/local/mysql|
|Zookeeper|3.4.6|/usr/local/zookeeper-3.4.6|
|Kafka|0.8.2.1|/usr/local/kafka_2.10.-0.8.2.1|
|Hadoop|2.6.5|/usr/local/hadoop-2.6.5|
|HBase|1.2.6|/usr/local/hbase-1.2.6|
|Hive|1.1.0|/usr/local/apache-hive-1.1.0-bin|
|MySQL JDBC|5.1.43||
|Scala|2.10.6|/usr/local/scala-2.10.6|
|Spark|1.6.3|/usr/local/spark-1.6.3-bin-hadoop2.6|
|Storm|1.1.1|/usr/local/apache-storm-1.1.1|
|Sqoop|1.4.6|/usr/local/sqoop-1.4.6|
|Pig|0.16.0|/usr/local/pig-0.16.0|

各节点搭载的服务为：

|Hostname|Services|
|-|-|
|cluster1|NameNode, DataNode, ZKFC, ResourceManager, NodeManager, HMaster, HRegionServer, Master, Worker, MySQL|
|cluster2|QuorumPeerMain, NameNode, DataNode, ZKFC, ResourceManager, NodeManager, HMaster, HRegionServer, Worker, MySQL|
|cluster3|QuorumPeerMain, DataNode, NodeManager, HRegionServer, Worker, Kafka, nimbus, core, logviewer|
|cluster4|QuorumPeerMain, DataNode, NodeManager, HRegionServer, Worker, Kafka, Supervisor, logviewer|
|cluster5|DataNode, NodeManager, HRegionServer, Worker, Kafka, Supervisor, logviewer|

**本系列教程中，命令以#开头的是需要使用root用户执行，$开头的使用普通用户（一般为hadoop用户）。而且本教程的操作系统是CentOS7，有些配置文件内的内容可能与Ubuntu等系统不符。**

# 关闭防火墙和Selinux
// 关闭防火墙和selinux
`# systemctl stop firewalld.service`

// 禁止firewall 开机启动
`# systemctl disable firewalld.service`

// 开机关闭Selinux，编辑Selinux配置文件
`# vi /etc/selinux/config`
将SELINUX设置为disabled
如下:
`SELINUX=disabled`
**千万别把SELINUXTYPE改了！**

// 重启
`# reboot`

// 重启机器后root用户查看Selinux状态
`# getenforce`

# 修改hosts文件
假设5台机器的IP地址分别为192.168.1.211-192.168.1.215
每台机器都要做如下修改：
// 修改hosts
`# vi /etc/hosts`
// 在最下面添加以下几行内容
```
192.168.1.211 cluster1
192.168.1.212 cluster2
192.168.1.213 cluster3
192.168.1.214 cluster4
192.168.1.215 cluster5
```
修改成这个样子后，对于这台机器来说，cluster1就代表了192.168.1.211，其他的也同理。

# ntp的安装与配置
一个集群内需要有一个机器运行ntp server，其他机器用ntpdate向它同步时间。Hbase和Spark都要求时间是严格同步的，所以ntp是必需的。
我们将ntp server设置在cluster1上，所以只在cluster1上面安装ntpserver，在其他机器上安装ntpdate。

ubuntu下使用如下命令安装
```
# apt-get install ntp
# apt-get install ntpdate
```

CentOS使用
```
# yum install ntp
# yum install ntpdate
```

配置时间服务器：

// cluster1上执行以下操作
`# vi /etc/ntp.conf`

注释掉以下4行，也就是在这4行前面加#
```
server 0.centos.pool.ntp.org iburst
server 1.centos.pool.ntp.org iburst
server 2.centos.pool.ntp.org iburst
server 3.centos.pool.ntp.org iburst
```
最下面加入以下内容，192.168.1.1和255.255.255.0分别为网关和掩码，127.127.1.0表示以本机时间为标准。
```
restrict default ignore
restrict 192.168.1.1 mask 255.255.255.0 nomodify notrap
server 127.127.1.0
```

保存后ubuntu使用`# /etc/init.d/ntp restart`重启ntp服务，CentOS使用`# service ntpd restart`
除了搭载ntp server的主机，其他所有机器，设定每天00:00向ntp server同步时间，并写入日志
`# crontab –e`
添加以下内容
```
0 0 * * * /usr/sbin/ntpdate cluster1>> /home/hadoop/ntpd.log
```
这样就完成了ntp的配置

// 手动同步时间，需要在每台机器上（除ntp server），使用`ntpdate cluster1`同步时间
`# ntpdate cluster1`

# 新建hadoop用户
每台机器上都要新建hadoop用户，这个用户专门用来维护集群，因为实际中使用root用户的机会很少，而且不安全。
// 新建hadoop组
`# groupadd hadoop`

// 新建hadoop用户
`# useradd -s /bin/bash -g hadoop -d /home/hadoop -m hadoop`

// 修改hadoop这个用户的密码
`# passwd hadoop`

# ssh密钥的生成与分发
ssh是Linux自带的服务，不需要安装。这里的目的是让节点间实现无密码登陆。其实就是当前机器生成密钥，然后用`ssh-copy-id`复制到其他机器上，这样这台机器就可以无密码直接登陆那台机器了。Hadoop主节点需要能无密码连接到其他的机器上。

在cluster1上，使用hadoop用户
```
// 使用hadoop用户
# su hadoop

// 切到home目录
$ cd ~/

// 生成密钥
ssh-keygen -t rsa
// 一路回车

//复制密钥
$ ssh-copy-id cluster1
yes
输入cluster的密码

$ ssh-copy-id cluster2
同上

$ ssh-copy-id cluster3
同上

$ ssh-copy-id cluster4
同上

$ ssh-copy-id cluster5
同上

// 然后测试能否无密码登陆
$ ssh cluster1
$ ssh cluster2
$ ssh cluster3
$ ssh cluster4
$ ssh cluster5
```

查看登陆时是否有密码，若无密码，则配置成功。
以上步骤需要在cluster2上也执行一遍，为了让cluster2也可以无密码登陆到其他机器上，因为cluster2也是主节点。

# jdk的安装与配置
安装hadoop集群，jdk是必须要装的，1.7和1.8都可以，不过从Hadoop3开始，好像只支持1.8+了，但是换成1.9和1.10会出问题，所以推荐用1.8，我这里用的是1.7。

将下载好后的jdk解压到/usr/local/下
`# vi /etc/profile`
将下面4行添加到环境变量中
```
export JAVA_HOME=/usr/local/jdk1.7.0_80  
export JRE_HOME=/usr/local/jdk1.7.0_80/jre 
export CLASSPATH=.:$JAVA\_HOME/lib:$JRE_HOME/lib:$CLASSPATH  
export PATH=$JAVA_HOME/bin:$JRE\_HOME/bin:$JAVA_HOME:$PATH
```
使用`# source /etc/profile`刷新环境变量
使用`# java -version`查看java版本验证是否安装成功，如果能看到Java的版本，说明安装成功，没有问题。