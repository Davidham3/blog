---
categories:
- 分布式平台
date: 2017-08-21 16:13:59+0000
description: zookeeper通常以“复制模式”运行于一个计算机集群上，这个计算机集群被称为一个“集合体”。zookeeper通过复制来实现高可用性，只要集合体中半数以上的机器处于可用状态，它就可以提供服务。出于这个原因，一个集合体通常包含奇数台机器。
draft: false
math: null
tags:
- Hadoop
- software
title: Hadoop HA安装三：zookeeper的安装
---
zookeeper通常以“复制模式”运行于一个计算机集群上，这个计算机集群被称为一个“集合体”。zookeeper通过复制来实现高可用性，只要集合体中半数以上的机器处于可用状态，它就可以提供服务。出于这个原因，一个集合体通常包含奇数台机器。
<!--more-->
# zookeeper的安装
zookeeper通常以“复制模式”运行于一个计算机集群上，这个计算机集群被称为一个“集合体”。zookeeper通过复制来实现高可用性，只要集合体中半数以上的机器处于可用状态，它就可以提供服务。出于这个原因，一个集合体通常包含奇数台机器。
## 安装
本文选择了在cluster2，cluster3和cluster4三台机器上安装
将zookeeper解压到/usr/local目录下，并配置环境变量
`# vi /etc/profile`
在最下面加上2行
```
export ZOOKEEPER_HOME=/usr/local/zookeeper-3.4.6
export PATH=$ZOOKEEPER\_HOME/bin:$PATH
```
然后在conf中新建zoo.cfg文件，输入以下内容：
```
# 客户端心跳时间(毫秒)
tickTime=2000
# 允许心跳间隔的最大时间
initLimit=10
# 同步时限
syncLimit=5
# 数据存储目录
dataDir=/home/hadoop_files/hadoop_data/zookeeper
# 数据日志存储目录
dataLogDir=/home/hadoop_files/hadoop_logs/zookeeper/dataLog
# 端口号
clientPort=2181
# 集群节点和服务端口配置
server.1=hadoop-cluster2:2888:3888
server.2=hadoop-cluster3:2888:3888
server.3=hadoop-cluster4:2888:3888
```
创建zookeeper的数据存储目录和日志存储目录
```
# mkdir -p /home/hadoop_files/hadoop_data/zookeeper
# mkdir -p /home/hadoop_files/hadoop_logs/zookeeper/dataLog
# mkdir -p /home/hadoop_files/hadoop_logs/zookeeper/logs
```

修改文件夹的权限
```
# chown -R hadoop:hadoop /home/hadoop_files
# chown -R hadoop:hadoop /usr/local/zookeeper-3.4.6
```

在cluster2号服务器的data目录中创建一个文件myid，输入内容为1，myid应与zoo.cfg中的集群节点相匹配
```
# echo "1" >> /home/hadoop_files/hadoop_data/zookeeper/myid
```

修改zookeeper的日志输出路径
`# vi bin/zkEnv.sh`
```
if [ "x${ZOO\_LOG\_DIR}" = "x" ]
then
   ZOO\_LOG\_DIR="/home/hadoop\_files/hadoop\_logs/zookeeper/logs"
fi
if [ "x${ZOO_LOG4J_PROP}" = "x" ]
then
   ZOO_LOG4J_PROP="INFO,ROLLINGFILE"
fi
```
修改zookeeper的日志配置文件
`# vi conf/log4j.properties`
```
zookeeper.root.logger=INFO,ROLLINGFILE
log4j.appender.ROLLINGFILE=org.apache.log4j.DailyRollingFileAppender
```

将这个zookeeper-3.4.6的目录复制到其他的两个节点上
```
# scp -r /usr/local/zookeeper-3.4.6 cluster3:/usr/local/
# scp -r /usr/local/zookeeper-3.4.6 cluster4:/usr/local/
```
复制后在那两台机器上使用root用户修改目录所有者为hadoop用户，并修改他们的myid为2和3。

退回hadoop用户
```
# exit
```

然后使用hadoop用户，使用`zkServer.sh start`分别启动三个zookeeper，顺序无所谓。三个都启动后，使用`jps`命令查看，若有QuorumPeerMain则说明服务正常启动，没有的话，使用`zkServer.sh start-foreground`查看一下哪里出了问题。

## 安装中遇到的问题
1. zookeeper启动不了
使用`zkServer.sh start-foreground`运行zookeeper，显示`line 131:exec java: not found`
解决办法：
    ```
    # cd /usr/local
    # chown –R hadoop:hadoop zookeeper-3.4.6
    ```
    改一下用户权限即可
2. 打开logs文件夹里面的zookeeper.log显示connection refused错误
原因：一般来说这是配置的问题，我出现这个问题的主要原因是，我在zoo.cfg中写了三个server，但是只在server1上启动zkServer.sh所以会出现connection refused。
事实上只在一个机器上启动zookeeper时，使用`zkServer.sh status`查看状态时，会显示zk可能没有运行，但是这并不是说明你的zookeeper有问题，只是那两个还没启动好而已，当3台机器的zookeeper都启动后，3台机器会自动进行投票，选出一个leader两个follower，此时再用`zkServer.sh status`查看状态的时候就可以看到这台机器是leader还是follower了。