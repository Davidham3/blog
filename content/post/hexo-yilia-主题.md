---
categories:
- 随笔
date: 2018-07-13 23:01:12+0000
description: 发现了一个特别好看的主题，yilia，但是已经不维护了，坑还挺多的，记录一下。
draft: false
math: null
tags:
- software
title: hexo yilia 主题
---
发现了一个特别好看的主题，yilia，但是已经不维护了，坑还挺多的，记录一下。
<!--more-->
主题地址：https://github.com/litten/hexo-theme-yilia
1. 首先git clone到themes目录中，记得要改名，目录名九教yilia，把前面的hexo-theme去掉

2. 修改完hexo根目录下_config.yml里面的theme为yilia后，运行服务器，发现出现
    ```
    <%- partial('_partial/head') %>
    <%- partial('_partial/header') %>
    <%- body %>
    <% if (theme.sidebar && theme.sidebar !== 'bottom'){ %> <%- partial('_partial/sidebar') %> <% } %>
    <%- partial('_partial/footer') %>
    <%- partial('_partial/mobile-nav') %> <%- partial('_partial/after-footer') %>
    ```
    说明少装了插件，执行以下命令安装插件：
    ```
    npm install hexo-renderer-ejs --save
    npm install hexo-renderer-stylus --save
    npm install hexo-renderer-marked --save
    ```

3. 进入文章后，头像就显示不出来了，这是源码中的bug。
    修改themes\yilia\layout\_partial\left-col.ejs的第六行，改为
    ```
    <img src="<%=theme.root%><%=theme.avatar%>" class="js-avatar">
    ```
    头像就设置为"img/avatar.jpg"即可。
    同时，还要修改themes\yilia\layout\_partial\mobile-nav.ejs
    里面的第10行，修改为
    ```
    <img src="<%=theme.root%><%=theme.avatar%>" class="js-avatar">
    ```

4. 打赏的二维码也有这个问题
    修改themes\yilia\layout\_partial\article.ejs
    找到<img class="reward-img"这个标签，改后面src的值
    支付宝的改成这个
    ```
    <%=theme.root%><%= theme.alipay%>
    ```
    微信的改成这个
    ```
    <%=theme.root%><%= theme.weixin%>
    ```

5. “所有文章”按钮的安装
    首先使用命令
    ```
    node -v
    ```
    检查版本是不是大于6.2
    然后在hexo的配置文件_config.yml最下面加上
    ```
    jsonContent:
        meta: false
        pages: false
        posts:
        title: true
        date: true
        path: true
        text: false
        raw: false
        content: false
        slug: false
        updated: false
        comments: false
        link: false
        permalink: false
        excerpt: false
        categories: false
        tags: true
    ```
    重启服务器即可

6. 分享功能有问题，我发现share_jia的微信分享不好使，就使用了Mob分享
    先去官网注册账号http://mob.com/，然后申请shareSDK，会得到一个App Key
    在yilia主题里面的_config.yml中的最后，加上
    ```
    sharesdk: true #是否开启分享
    shareSDKappkey: 你的App Key
    ```
    然后在layout中的_partial中新建目录share，
    创建文件：yilia/layout/_partial/share/share.ejs
    ```html
    <!--MOB SHARE BEGIN-->
    <div class="-mob-share-ui-button -mob-share-open">分享</div>
    <div class="-mob-share-ui" style="display: none">
        <ul class="-mob-share-list">
            <li class="-mob-share-weibo"><p>新浪微博</p></li>
            <li class="-mob-share-tencentweibo"><p>腾讯微博</p></li>
            <li class="-mob-share-qzone"><p>QQ空间</p></li>
            <li class="-mob-share-qq"><p>QQ好友</p></li>
            <li class="-mob-share-weixin"><p>微信</p></li>                        <li class="-mob-share-twitter"><p>Twitter</p></li>     
            <li class="-mob-share-youdao"><p>有道云笔记</p></li>
            <li class="-mob-share-mingdao"><p>明道</p></li>            
            <li class="-mob-share-linkedin"><p>LinkedIn</p></li>
        </ul>
    <div class="-mob-share-close">取消</div>
    </div>
    <div class="-mob-share-ui-bg"></div>
    <script id="-mob-share" src="http://f1.webshare.mob.com/code/mob-share.js?appkey={{ theme.shareSDKappkey }}"></script>
    <!--MOB SHARE END-->
    ```
    然后编辑layout/_partial/article.ejs
    找个合适的位置加入以下内容
    ```
    <% if (!index && theme.sharesdk){ %>
        <%- partial('_partial/share/share.ejs') %>
    <% } %>
    ```

7. 左边昵称的字体有点丑，在themes\yilia\source\main.0cf68a.css里面修改，找到header-author，修改里面的font-family，我改成了
    ```css
    font-family:"Times New Roman",Georgia,Serif
    ```

8. 之前的那个share_jia我修复了，之前微信分享之所以不成功，好像是因为百度网盘取消了生成二维码的功能，导致之前的链接不可用了。解决方法是修改themes\yilia\layout\_partial\post\share.ejs
    把第49行中的`//pan.baidu.com/share/qrcode?url=`修改为
    ```
    //api.qrserver.com/v1/create-qr-code/?size=150x150&data=
    ```

9. 如何在左侧显示总文章数？
    修改themes\yilia\layout\_partial\left-col.ejs
    在
    ```
    <nav class="header-menu">
        <ul>
        <% for (var i in theme.menu){ %>
            <li><a href="<%- url_for(theme.menu[i]) %>"><%= i %></a></li>
        <%}%>
        </ul>
    </nav>
    ```
    后面加入
    ```
    <nav>
        总文章数 <%=site.posts.length%>
    </nav>
    ```

10. 添加评论系统
    yilia默认带了几个系统，但我是从next这个主题转过来的，之前用的是来必力(livere)，不想换了，就得手动在yilia里面加。
    首先是去注册livere，然后获取到自己的id
    新建themes\yilia\layout\_partial\comment\livere.ejs
    内容如下：
    ```
    <!-- 来必力City版安装代码 -->
    <div id="lv-container" data-id="city" data-uid="<%=theme.livere_uid%>">
    <script type="text/javascript">
        (function(d, s) {
            var j, e = d.getElementsByTagName(s)[0];

            if (typeof LivereTower === 'function') { return; }

            j = d.createElement(s);
            j.src = 'https://cdn-city.livere.com/js/embed.dist.js';
            j.async = true;

            e.parentNode.insertBefore(j, e);
        })(document, 'script');
    </script>
    <noscript>为正常使用来必力评论功能请激活JavaScript</noscript>
    </div>
    <!-- City版安装代码已完成 -->
    ```

    然后编辑themes\yilia\layout\_partial\article.ejs
    找到这句话：`<% if (!index && post.comments){ %>`
    在下面直接加入：
    ```
    <% if (theme.livere){ %>
    <%- partial('comment/livere', {
    key: post.slug,
    title: post.title,
    url: config.url+url_for(post.path)
    }) %>
    <% } %>
    ```
    意思就是如果主题配置文件中有livere这个变量，且不为false，那就在下面加入comment/livere.ejs里面的内容
    所以接下来需要在主题配置文件(themes\yilia\_config.yml)中添加以下内容：
    ```
    livere: true
    livere_uid: 你的id
    ```

11. 添加字数统计
    首先需要安装[hexo-wordcount](https://github.com/willin/hexo-wordcount)
    使用如下命令安装
    ```
    npm i --save hexo-wordcount
    ```
    # Node 版本7.6.0之前,请安装 2.x 版本 (Node.js v7.6.0 and previous)
    ```
    npm install hexo-wordcount@2 --save
    ```
    然后在themes\yilia\layout\_partial\left-col.ejs中添加
    ```
    总字数 <span class="post-count"><%= totalcount(site, '0,0.0a') %></span>
    ```
    编辑themes\yilia\layout\_partial\article.ejs
    在header下面加入
    ```
    <div align="center" class="post-count">
        字数：<%= wordcount(post.content) %>字 | 预计阅读时长：<%= min2read(post.content) %>分钟
    </div>
    ```
    即可显示单篇字数和预计阅读时长。

12. 关于访问litten.me:9005的问题，这个主题的作者之前为了更好地完善这个主题，有时候会收集用户的客户端信息，详情请见[https://github.com/litten/hexo-theme-yilia/issues/528](https://github.com/litten/hexo-theme-yilia/issues/528)，如果不想被统计，就将themes\yilia\source-src\js\report.js里面的内容清空，不过这个请求是异步的，不会影响博客加载速度，而且作者已经不维护了，所以关不关都行。