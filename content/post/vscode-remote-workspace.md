---
categories:
- software
date: 2018-06-15 09:46:40+0000
description: 使用vscode管理远程服务器上的文件与项目。
draft: false
math: null
tags:
- software
- vscode
title: vscode-remote-workspace
---
使用vscode管理远程服务器上的文件与项目。
<!--more-->

[vscode-remote-workspace](https://github.com/mkloubert/vscode-remote-workspace)是一个vscode中的插件，可以管理远程存储上的文件、项目，还可以执行命令。
支持的系统很多：
1. Auzre
2. Dropbox
3. FTP
4. FTPs
5. S3 Buckets
6. SFTP
7. Slack
8. WebDAV

以SFTP为例，只要写这么一个配置文件即可
```
{
    "folders": [{
        "uri": "sftp://my-user:my-password@sftp.example.com/",
        "name": "My SFTP folder"
    }],
    "settings": {}
}
```

举个例子：
```
{
    "folders": [{
        "uri": "sftp://Davidham3:my-password@my-linux-server-ip/data/Davidham3",
        "name": "My SFTP folder"
    }],
    "settings": {}
}
```
保存成名为my-linux-server.code-workspace的文件后，右键点击这个文件，使用vscode打开即可。或是打开vscode后，点击“文件”，选择“打开工作区”，然后选择这个文件即可。
![](/images/vscode-remote-workspace/demo1.gif)
使用F1，然后输入`execute remote command`，然后就可以输入命令，直接在远程机器上运行。
![](/images/vscode-remote-workspace/demo2.PNG)

# 安装方法
打开vscode后，选择左侧第五个按钮，进入商店，然后查找vscode-remote-workspace，点击绿色的安装按钮安装即可，安装后点蓝色的“重新加载”按钮即可。
![](/images/vscode-remote-workspace/demo3.PNG)

# 问题
不过使用execute remote command的时候，如果程序可以正常运行，不报错，那这个工具是可以显示内容的，但是一旦程序出错了，就不会有任何错误信息显示。这点这个工具没法处理。所以解决方案就是，直接用下面的终端，ssh进去。最新版本的Windows10已经内置了OpenSSH，所以直接用`ssh 用户名@hostname`就可以连接到服务器，然后执行命令跑程序。