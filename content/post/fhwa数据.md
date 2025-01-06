---
categories:
- dataset
date: 2018-07-12 14:05:06+0000
description: FHWA数据数据地址：[https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/](https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/)数据包含了2011到2016年的全年数据。数据说明：[https://www.fhwa.dot.gov/policyinformation/tmguide/](https://www.fhwa.dot.gov/policyinformation/tmguide/)说明共有三版，但是数据对应的其实是最老的一个版本，2001年修订的那个文档。
draft: false
math: true
tags:
- dataset
title: FHWA数据
---
FHWA数据
数据地址：[https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/](https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/)
数据包含了2011到2016年的全年数据。
数据说明：[https://www.fhwa.dot.gov/policyinformation/tmguide/](https://www.fhwa.dot.gov/policyinformation/tmguide/)
说明共有三版，但是数据对应的其实是最老的一个版本，2001年修订的那个文档。
<!--more-->

本文只对文档中的一部分进行说明，详细的还是得看官方文档。

# 数据概况
数据是美国高速公路局收集的交通流量数据，每月一组。

数据的格式是FHWA定的，数据会被输入到FHWA维护的两个数据库中，Traffic Volume Trends（TVT）和Vehicle Travel Information System（VTRIS）。TVT系统处理连续的流量数据，按月生成交通流量趋势报告。VIRIS系统处理车辆分类和卡车称重数据，作为年度卡车重量研究。这两个数据管理系统处理、验证、汇总、维护交通数据。TVT和VIRIS向任何人提供，通过这个地址[http://www.fhwa.dot.gov/ohim/tvtw/tvtwpage.htm](http://www.fhwa.dot.gov/ohim/tvtw/tvtwpage.htm)获得。数据收集计划由the Office of Management and Budget批准，OMB # 2125-0587，期限是2004年4月30日。（因为现在看的文档是2001年编写的，所以已经过时了，但是数据格式没有过时，新版的文档反而和当前数据(2016年)的格式对不上）

数据记录分为四类，站点描述数据、流量数据、车辆分类数据、卡车重量数据。每类数据都有自己的格式。接下来会分章节讨论四类数据的格式规范。

注意：一些字段被标记为"critical"，意思是必不可少的字段。这里描述的所有数据都上ASCII flat文件。对于缺失的字段或数据，会用空格代替。数字是右对其的，如果没有说明的话左侧补空格或0。数值型字段缺失或不可用的数据，输入空格或右对齐的-1。

四类数据记录中的部分数据项是相同的。举个例子，所有的记录包含一个6字符长度的站点标识符。这使得每个州都需要使用一个共同的标识系统。

站点描述中的一些字段被替换成了需要与GIS连接起来的交通数据。这会使得数据与NHPN(National Highway Planning Network)或相似的系统重叠。

# 站点数据描述
站点数据描述对所有的流量、车辆分类、称重站点都适用。一个站点描述文件包含了对每个交通监测站的记录（每年）。所有的字段都是字符型。命名规则是"ssyy.STA"（现在已经不是了）。

2016年数据的文件名：AK_2016 (TMAS).STA，前两个字符表示这个州的缩写。TMAS是指Travel Monitoring Analysis System(National)。

下标|含义|样例|critical/optional
-|-|-|-
1|Record Type: 记录类型(S表示站点)|S|c
2-3|FIPS State Codes: 州编号|02|c
4-9|Station Identification: 站标识符|000101|c
10|Direction of Travel Code: 路的走向|1|c
11|Lane of Travel: 哪条路|1|c
12-13|Year of Data: 年|16|c
14-15|Functional Classification Code: 功能类型|1R|o
16|Number of Lanes in Direction Indicated: 这个方向几条路|1|o
17|Sample Type for Traffic Volume: 是否用于监测流量|T|o
18|Number of Lanes Monitored for Traffic Volume: 几条路监测流量|1|o
19|Method of Traffic Volume Counting: 流量监测方法|3|o
20|Sample Type for Vehicle Classification: 车辆分类方法| |o
21|Number of Lanes Monitored for Vehicle Classification: 几条路对车辆分类|0|o
22|Method of Vehicle Classification: 车辆分类方法|0|o
23|Algorithm for Vehicle Classification: 车辆分类算法| |o
24-25|Classification System for Vehicle Classification: 车辆分类系统|13|o
26|Sample Type for Truck Weight: 称重类型| |o
27|Number of Lanes Monitored for Truck Weight: 几条路称重|0|o
28|Method of Truck Weighing: 称重方法|0|o
29|Calibration of Weighing System: 称重系统精度| |o
30|Method of Data Retrieval: 数据获得的类型|2|o
31|Type of Sensor: 检测器类型|P|o
32|Second Type of Sensor: 检测器的第二种类型|L|o
33|Primary Purpose: 安装检测器的意图|P|o
34-45|LRS Identification|001700000000|o
46-51|LRS Location Point| 81967|o
52-59|Latitude: 纬度|62351650|o
60-68|Longitude: 经度|150252360|o
69-72|SHRP Site Identification|    |o
73-78|Previous Station ID|      |o
79-80|Year Station Established: 哪年建的站点|91|o
81-82|Year Station Discontinued|00|o
83-85|FIPS County Code|170|o
86|HPMS Sample Type|Y|o
87-98|HPMS Sample Identifier|170000008007|o
99|National Highway System|Y|o
100|Posted Route Signing|3|o
101-108|Posted Signed Route Number|00000003|o
109|Concurrent Route Signing|0|o
110-117|Concurrent Signed Route Number|      |  o
118-167|Station Location|PARKS HIGHWAY AT CHULITNA - NB |o

数据样例：
S0200010111161R1T13 00 13 00 2PLP001700000000 8196762351650150252360          9100170Y170000008007Y3000000030        PARKS HIGHWAY AT CHULITNA - NB                    (后面有很多空格)

# 流量数据格式
2016年数据的文件名：AK_JAN_2016 (TMAS).VOL
12个字段

下标|描述|样例|critical/optional
-|-|-|-
1|Record Type: 记录类型|3|c
2-3|FIPS State Code|02|c
4-5|Functional Classification Code|1R|c
6-11|Station Identification|000101|c
12|Direction of Travel Code|1|c
13|Lane of Travel|1|c
14-15|Year of Data|16|c
16-17|Month of Data|01|c
18-19|Day of Data|01|c
20|Day of Week(1表示周日)|6|o
21-25,...136-140|Traffic Volume Counted Fields(从00:00到24:00，24个小时的流量)|00005 00004 00002 ... 00003|o
141|Restrictions(1表示施工或活动影响流量，2表示检测器出现问题，0表示正常)|0|o

数据样例：
3021R0001011116010160000500004000020000000001000010000200001000150003100026000430003200052000340002800024000140001400007000120000800007000030
不知道为什么这行数据在浏览器里面直接出去了。。。

关于车辆分类和称重数据就不描述了，可以直接看官方文档。