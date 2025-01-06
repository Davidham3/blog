---
categories:
- 分布式平台
date: 2017-08-21 16:13:31+0000
draft: false
math: null
tags:
- Hadoop
- software
title: Hadoop HA安装二：MySQL双机热备
---
Hadoop HA安装二：MySQL双机热备

<!--more-->

## 安装MySQL
安装前先安装一下MySQL的依赖
ubuntu:
```
# apt-get install libaio-dev
```

CentOS:
```
# yum install libaio
```

看了很多教程都不靠谱。。。还是官方教程最靠谱：
[Installing MySQL on Unix/Linux Using Generic Binaries](https://dev.mysql.com/doc/refman/5.6/en/binary-installation.html)

下载mysql-5.6.37-linux-glibc2.12-x86_64
`# cp mysql-5.6.37-linux-glibc2.12-x86_64.tar.gz /usr/local/`

解压到/usr/local/
`# tar -zxvf mysql-5.6.37-linux-glibc2.12-x86_64.tar.gz`

改名为mysql
`# mv mysql-5.6.37-linux-glibc2.12-x86_64 mysql`

删除安装包
`# rm mysql-5.6.37-linux-glibc2.12-x86_64.tar.gz`

修改环境变量
`# vi /etc/profile`
在最下面添加
```
export MYSQL_HOME=/usr/local/mysql
export PATH=$MYSQL_HOME/bin:$PATH
```

新建用户和用户组：mysql
`# groupadd mysql`
`# useradd -r -g mysql -s /bin/false mysql`

`# cd /usr/local/mysql`

修改目录的拥有者
`# chown -R mysql .`(**重要！**)
`# chgrp -R mysql .`(**重要！**)

安装MySQL
`# scripts/mysql_install_db --user=mysql`

修改当前目录拥有者为root用户
`# chown -R root .`

修改当前data目录拥有者为mysql用户
`# chown -R mysql data`

启动MySQL进程
`# bin/mysqld_safe --user=mysql &`

此时这个窗口会卡住，新建一个terminal，进入/usr/local/mysql中

进入mysql控制台
`# bin/mysql`

退出
`exit;`

进行MySQL的root用户密码的修改等操作
`# ./bin/mysql_secure_installation`
首先要求输入root密码，由于我们没有设置过root密码，括号里面说了，如果没有root密码就直接按回车。是否设定root密码，选y，设定密码为cluster，是否移除匿名用户：y。然后有个是否关闭root账户的远程登录，选n，删除test这个数据库？y，更新权限？y，然后ok。

`# cp support-files/mysql.server /etc/init.d/mysql.server`

查看MySQL的进程号
`# ps -ef | grep mysql`

如果有的话就kill掉，保证MySQL已经中断运行了，一般kill掉/usr/local/mysql/bin/mysqld开头的即可
`# kill 进程号`

启动MySQL
`# /etc/init.d/mysql.server start -user=mysql`
`# exit`


还需要配置一下访问权限：
```
$ mysql -u root -p
mysql> GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' IDENTIFIED BY 'cluster' WITH GRANT OPTION;
mysql> FLUSH PRIVILEGES;
```

这样就可以使用root用户在任意地点登陆，而不再限于localhost了
关于服务的三个命令。
启动mysql：`# /etc/init.d/mysql.server start -user=mysql`
停止mysql：`# mysqladmin -u root -p shutdown`

## 修改MySQL的数据存储位置
在装系统进行分区的时候，有些时候会创建比较大的分区开存数据，我们可以将MySQL的数据存放到这个区内，假设这个区为/file0

`$ su root`

把MySQL服务进程停掉：
`# mysqladmin -u root -p shutdown`

新建新的dataDir
`# mkdir /file0/mysql_data`

把/usr/local/mysql/data里面的东西移到/file0/mysql_data下
`# cp /usr/local/mysql/data/* /file0/mysql_data`

编辑MySQL的配置文件/etc/my.cnf
如果没有的话，就把/usr/local/mysql里面的my.cnf复制过去
`# cp /usr/local/mysql/my.cnf /etc/`
`# vi /etc/my.cnf`

把里面的basedir, datadir, port修改成下面的内容
```
basedir=/usr/local/mysql
datadir=/file0/mysql_data
port=3306
```

修改MySQL启动脚本/etc/init.d/mysql.server
`# vi /etc/init.d/mysql.server`
把里面的basedir和datadir作如上修改

修改新目录的权限：
`# chown –R mysql /file0/mysql_data`
`# chgrp –R mysql /file0/mysql_data`
退出root用户
重新启动MySQL服务
`$ /etc/init.d/mysql.server start –user=mysql`

进入mysql
`$ mysql –u root -p`

查看目录是否已经更改
`mysql> show variables like “datadir”;`

## 高可用的MySQL双机热备安装教程
开启二进制日志，设置id
`# vi /etc/my.cnf`
```
[mysqld]
server-id = 1                                 #backup这台设置2
log-bin = mysql-bin
binlog-ignore-db = mysql,information_schema   #忽略写入binlog日志的库
auto-increment-increment = 2                  #字段变化增量值
auto-increment-offset = 1                     #初始字段ID为1
slave-skip-errors = all                       #忽略所有复制产生的错误    
```

重启MySQL服务
`# mysqladmin -u root -p shutdown`
`# /etc/init.d/mysql.server start –user=mysql`

先查看下log bin日志和pos值位置
里面有个File和Position，分别是log_file和log_pos的值，一会儿要填
`mysql> show master status;`

master配置如下：
```
mysql> GRANT REPLICATION SLAVE ON *.* TO 'replication'@'192.168.1.%' IDENTIFIED  BY 'replication';
mysql> flush privileges;
mysql> change master to
    ->  master_host='192.168.1.212',            # 此处输入slave的ip地址
    ->  master_user='replication',
    ->  master_password='replication',
    ->  master_log_file='mysql-bin.000001',
    ->  master_log_pos=120;                    #对端状态显示的值
mysql> start slave;                            #启动同步
```

backup配置如下：
```
mysql> GRANT REPLICATION SLAVE ON *.* TO 'replication'@'192.168.1.%' IDENTIFIED BY 'replication';
mysql> flush privileges;
mysql> change master to
    ->  master_host='192.168.1.211',        # 此处输入master的ip地址
    ->  master_user='replication',
    ->  master_password='replication',
    ->  master_log_file='mysql-bin.000001',
    ->  master_log_pos=120;
mysql> start slave;
```

MySQL双击热备安装完成

## 测试
在一台机器上建立一个数据库，创建一个表，在另一台机器上查询是有结果的，说明安装成功。