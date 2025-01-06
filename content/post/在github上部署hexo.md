---
categories: 随笔
date: 2018-02-20 16:53:50+0000
draft: false
math: null
tags:
- software
title: 在GitHub上部署hexo博客
---

在GitHub Pages部署hexo博客
<!--more-->

# 需要的工具
1. node.js
2. git

# 安装及部署
1. 安装完node.js和git后安装hexo  
`npm install hexo -g`
安装后使用`hexo -v`查看版本号，看是否安装成功
2. 创建hexo项目
找个文件夹作为博客的目录
在这个目录下使用
`hexo init`
初始化该目录
3. 使用`npm install`安装需要的组件
4. 使用`npm install hexo-deployer-git --save`安装插件
5. 使用`hexo generate`或`hexo g`生成当前的博客
6. 使用`hexo server`或`hexo s`启动服务器
然后就可以打开浏览器访问localhost:4000在本地查看当前的博客
7. 生成SSH密钥
打开Git Bash，使用以下命令配置git
`git config --global user.name "你的github用户的名字"`
`git config --global user.email "你的github账户邮箱"`
`cd ~/.ssh`
`ssh-keygen -t rsa -C "你的github账户邮箱"`
连续三个回车
`eval "$(ssh-agent -s)"`，添加密钥到ssh-agent
`ssh-add ~/.ssh/id_rsa`，添加生成的SSH key到ssh-agent
`cat ~/.ssh/id_rsa.pub`
复制此时显示的内容，内容应该是以ssh-rsa开头
8. Ctrl+C退出后，在GitHub上新建一个新的仓库，仓库名随意，不过需要记录下来，我这里起名叫blog，最下面的Initialize this repository with a README要勾选上，然后保存即可。进入这个仓库后选择Settings，在左侧选项卡Options中翻到下面，GItHub Pages这项，Source选择master branch，选择save后，会在这部分的标题处写明这个仓库的url，这就是你博客的url了。还是页面的左侧的选项卡，Deploy 选择Add deploy key，添加密钥。
Title随意，我设置为了blog
Key粘贴我们刚才复制的那一段。
最下面Allow write access要打勾.
选择Add Key即可。
然后在Git Bash中使用
`ssh -T git@github.com`测试，如果看到Hi后面是你的用户名，就说明成功了。
9. 修改hexo配置文件
打开本地博客的根目录，找到_config.yml文件，
在文件的开头处，第二部分，URL这部分改成如下内容：
```
# URL
## If your site is put in a subdirectory, set url as 'http://yoursite.com/child' and root as '/child/'
url: http://yoursite.com/blog
root: /blog/
permalink: :year/:month/:day/:title/
permalink_defaults:
```
  这里的url和root这两项都需要修改。url在后面要加仓库名，我的仓库叫blog，所以写成了`http://yoursite.com/仓库名`，同理root要修改成`/仓库名/`。
  在文件的结尾处，Deployment这部分改成如下内容：
  ```
# Deployment
## Docs: https://hexo.io/docs/deployment.html
deploy:
  type: git
  repository: git@github.com:Davidham3/blog.git
  branch: master
  ```
  需要注意的是，这里的repository这项，应该去GitHub里面你新建的那个叫blog的仓库里面找。进入仓库主页后，点击右侧绿色的按钮Clone or download，在新弹出的窗口右上角选择Use SSH，然后将下面的文字复制粘贴到此处。
修改完配置文件后保存退出即可。
10. 使用`hexo clean`清除缓存
11. 使用`hexo g`生成博客
12. 使用`hexo deploy`或`hexo d`将博客部署至GitHub上，打开刚才GitHub Pages设置里面给出的url，就可以进入你的博客了。以上两步也可以连写为`hexo d -g`。