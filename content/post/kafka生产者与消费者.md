---
categories:
- 分布式平台
date: 2018-07-05 20:54:26+0000
draft: false
math: null
tags:
- Kafka
title: Kafka生产者与消费者
---
Kafka是一个分布式、流式消息平台，是一套发布订阅系统，通俗来说就是Kafka producer发布数据至Kafka brokers，然后由Kafka consumer从brokers拉取数据，进行消费。
<!--more-->
最近上数据仓库的课，学习了Kafka的使用方式以及Kafka的原理。

Kafka官网：[Apache Kafka](http://kafka.apache.org/)

Kafka是一个分布式、流式消息平台，是一套发布订阅系统，通俗来说就是Kafka producer发布数据至Kafka brokers，然后由Kafka consumer从brokers拉取数据，进行消费。

**日志**
有意思的特性是Kafka内的数据都是以日志的形式存储，即便消费完也不会消失，配置文件中配置了过了多长时间日志会销毁掉。这样设计的好处有很多，consumer是有group的，每个组进行消费的时候，都会有个偏移量offset记录在zookeeper中，通过这个offset就知道下次从哪里开始消费了，不同组的offset不一样，这样每个组都可以按照自己的需要进行消费。

**主题**
Kafka的记录是有主题的，这样producer发送到broker的数据其实就是打上了标签，有了分类，消费的时候可以按主题消费，相当于一开始就用主题对数据进行了区分。

**效率**
Kafka集群同时也作为缓冲区，平衡producer和consumer两边的工作进度，不会因为一方过慢造成阻塞一类的问题。

**语言**
写起来的话，肯定是java和scala最好，因为Kafka就是由这两种语言编写的，当然，也有其他语言的接口，比如python。python的话比较有意思的是有两个Kafka框架，一个是[kafka-python](https://kafka-python.readthedocs.io/en/master/index.html)，另一个是[pykafka](http://pykafka.readthedocs.io/en/latest/#)。推荐使用后者，前者在创建consumer group的时候不是很方便，group内的每个consumer消费的内容都一样，没有实现去重与平衡，这些都需要自己实现，后者的balanced_consumer就挺好的。

[自己写的例子](https://github.com/Davidham3/pykafka_examples)