---
categories:
- dataset
date: 2018-07-11 15:54:11+0000
description: MnDOT的全称是Minnesota Department of Transportation。RTMC traffic data是其的一个子集。美国明尼苏达州双子城交通管理中心的交通数据。地址：[http://www.d.umn.edu/~tkwon/TMCdata/TMCarchive.html](http://www.d.umn.edu/~tkwon/TMCdata/TMCarchive.html)
draft: false
math: true
tags:
- dataset
title: MnDOT traffic data
---
MnDOT的全称是Minnesota Department of Transportation。RTMC traffic data是其的一个子集。美国明尼苏达州双子城交通管理中心的交通数据。
地址：[http://www.d.umn.edu/~tkwon/TMCdata/TMCarchive.html](http://www.d.umn.edu/~tkwon/TMCdata/TMCarchive.html)
<!--more-->
数据是RTMC采集的连续数据，是MnDOT的一个子集，超过4500个每30秒为间隔的线圈检测器部署Twin Cities Metro freeways。最近加入了Rochester线圈数据。每天的数据都会UMD的服务器被打包进一个zip文件，之后存入这个仓库。文件名是"yyyymmdd.traffic"，分别是年月日。使用unzip软件直接解压即可。解压后有9000个文件，4500个是流量数据，文件名是"###.v30"，另外4000个文件是占用率文件，是"###.o30"或者"###.c30"。###表示检测器的id。数据服务由UMD的Transportation Research Data Lab(TDRL)提供，旨在与学者分享资源与思路。我们鼓励数据使用者与我们联系，分享研究成果与想法。这个数据是免费的，但是禁止用于商业用途。最后，感谢RTMC如此慷慨地提供交通数据。

数据格式：
[http://www.d.umn.edu/~tkwon/TMCdata/Traffic.html](http://www.d.umn.edu/~tkwon/TMCdata/Traffic.html)
MN/DOT已经收集了路中的检测器的数据很多年了。从2000年3月开始，在Twin Cities metro area，超过4000个检测器每30秒都会收集一次数据。原始数据包括了流量和占有率。每天都会有大量的数据，将这些数据存储进传统数据库的价值比乱放大很多。因此，这些数据的存储促使了MN/DOT交通数据文件格式的发展。这个格式现在是TDRL的UTSDF的一个特例。

UTSDF的优点很多。最重要的好处就是简单。早期的文件格式有复杂的bit操作，对数据分析工具很难操作。后来所有的数据存成8bit或16bit的整数解决了这个问题。这个格式的另一个好处是它的紧凑性。早期的格式，数据33M。现在这个格式，同样的数据只有13M（精度不变）。早期格式的另一个问题是30秒、5分钟的区分使得获取数据很麻烦。现在这个数据把所有数据融合到一个文件中，简化了读取数据的过程。

另一个重要的好处是可扩展性，未来可能在不牺牲紧凑性的情况下增加其他类型的数据（比如速度）。

每个traffic数据文件包含了一天的交通数据。文件一版命名为8个数字的日期加.traffic的后缀。压缩成了zip格式。每个检测器有两个文件，一个是整天的流量，另一个是占用率。这些文件的命名是检测器的id。流量的后缀是.v30，占用率是.o30。所以如果有个编号为100的检测器，那就有两个文件，100.v30和100.o30。

流量文件（.v30）共2880个字节。每字节是一个8比特带符号的流量值，每天30秒为一个周期。-1表示缺失值。最开始的8bit表示一天最开始的值，也就是午夜0点0分30秒，最后一个值是11点59分30秒。

占用率文件（.o30）和流量文件很相似，除了每个值是16bit。每个文件是5760字节。占用率值是从0到1000（百分点的十分之一）的fixed-point interger。-1表示缺失，16bit是高位优先（high-byte first order）。

以上格式说明修订于: 23 March 2000

附录：2001年8月3日

1. .c30文件是记录在"scans"中，并且比.o30文件更精确。不久所有的数据都会使用.c30格式。Scans定义为$\frac{1}{60}$秒，所以数据的范围是0到1800（30秒 $\times$ 60 scans/second），老版的文件.o30表示的是千分之一为单位的占有率，所以范围是0到1000。这是这两个文件的区别。如果你想要0到100的数据，将scan数据除以18，或者将占用率数据除以10。任何在这个范围外的数据都是有问题的数据。

2. 对于流量数据，把他当成有符号或者没符号无所谓。因为样本是30秒的流量数据，如果有40量车通过那就说明每小时会通过4800量车，平均车与车之间差了0.75秒。肯定是不可能，所以我建议如果数据不在0到40之间，那就说明是异常值。

3. 那些不同的负数是数据采集软件的小bug。未来我们会修复他们，所以对于流量数据，任何不在合理范围的数据都应该被当成异常值。

以上格式信息由TMC Mn/DOT的Doug Lau提供。