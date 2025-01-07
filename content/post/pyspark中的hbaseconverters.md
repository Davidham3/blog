---
categories:
- 分布式平台
date: 2019-04-06 16:43:29+0000
description: 最近项目上有个需求，使用 pyspark 读取 HBase 中存储的 java.math.BigDecimal。
draft: false
math: null
tags:
- Spark
- software
title: pyspark中的HBaseConverters
---
最近项目上有个需求，使用 pyspark 读取 HBase 中存储的 java.math.BigDecimal。

<!--more-->

最近甲方让我们写一个 pyspark 的教程，他们以后打算使用 pyspark 开发。他们的数据是那种精度要求比较高的数据，我们使用 java.math.BigDecimal 表示数字，然后转成 byte[] 后存入了 HBase，但是 python 是没法直接读取这个 BigDecimal，所以需要使用 spark-examples 中 HBaseConverters.scala 读取。

我们讨论的 spark 版本是 1.6，因为用的是 CDH 5，所以是这个版本。

原理实际上是，pyspark 在读取 HBase 的时候需要借助 org.apache.spark.examples.pythonconverters 这么一个类，这个类实际上是 scala 将 HBase 中的数据读取后，转换成 json 字符串返回，这样 pyspark 可以通过这个类从 HBase 中直接获取到 json 字符串这样的返回值。

可以从 [HBaseConverters.scala](https://github.com/apache/spark/blob/branch-1.6/examples/src/main/scala/org/apache/spark/examples/pythonconverters/HBaseConverters.scala) 这里看到 HBaseConverters.scala 的源码，我们感兴趣的是从 HBase 中查询 value 这一部分：

```scala
package org.apache.spark.examples.pythonconverters

import scala.collection.JavaConverters._
import scala.util.parsing.json.JSONObject

import org.apache.spark.api.python.Converter
import org.apache.hadoop.hbase.client.{Put, Result}
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.hbase.KeyValue.Type
import org.apache.hadoop.hbase.CellUtil

/**
 * Implementation of [[org.apache.spark.api.python.Converter]] that converts all
 * the records in an HBase Result to a String
 */
class HBaseResultToStringConverter extends Converter[Any, String] {
  override def convert(obj: Any): String = {
    val result = obj.asInstanceOf[Result]
    val output = result.listCells.asScala.map(cell =>
        Map(
          "row" -> Bytes.toStringBinary(CellUtil.cloneRow(cell)),
          "columnFamily" -> Bytes.toStringBinary(CellUtil.cloneFamily(cell)),
          "qualifier" -> Bytes.toStringBinary(CellUtil.cloneQualifier(cell)),
          "timestamp" -> cell.getTimestamp.toString,
          "type" -> Type.codeToType(cell.getTypeByte).toString,
          "value" -> Bytes.toStringBinary(CellUtil.cloneValue(cell))
        )
    )
    output.map(JSONObject(_).toString()).mkString("\n")
  }
}
```

这段代码很简单，实际上就是使用 java HBase 的 API 读取 HBase 中的值，将所有的值转换为 String 返回，我需要做的，只是将 value 这个字段的值，先从 byte[] 转到 BigDecimal，再转换为 String 即可。

```scala
class MyHBaseResultToStringConverter extends Converter[Any, String] {
  override def convert(obj: Any): String = {
    val result = obj.asInstanceOf[Result]
    val output = result.listCells.asScala.map(cell =>
        Map(
          "row" -> Bytes.toStringBinary(CellUtil.cloneRow(cell)),
          "columnFamily" -> Bytes.toStringBinary(CellUtil.cloneFamily(cell)),
          "qualifier" -> Bytes.toStringBinary(CellUtil.cloneQualifier(cell)),
          "timestamp" -> cell.getTimestamp.toString,
          "type" -> Type.codeToType(cell.getTypeByte).toString,
          "value" -> Bytes.toBigDecimal(CellUtil.cloneValue(cell)).toString()
        )
    )
    output.map(JSONObject(_).toString()).mkString("\n")
  }
}
```

这样代码就改完了，然后需要编译，打成 jar 包。

装好 maven，我装的是 3.6.0，不需要配置什么。

下载 spark 的源码，最开始我从 github 上面下载的，发现速度很慢，然后就去 spark 官网，找仓库中的源代码下载下来。

编译 spark-examples 的时候需要先从根目录中把 scalastyle-config.xml 拷贝到 examples 目录下再进行编译

cd 到 examples 目录下，使用以下命令编译 spark-examples

```
mvn clean install -pl :spark-examples_2.10
```

![Figure1](/blog/images/pyspark中的hbaseconverters/Fig1.JPG)

编译的时候没有遇到错误，编译好的包在同级目录下的 target 中，有个叫 spark-examples_2.10-1.6.0.jar 的文件。

然后就是使用这个包读取 HBase 中的 BigDecimal了：

我们使用 standalone 模式运行 pyspark：
```
pyspark --master spark://host1:7077 --jars spark-examples_2.10-1.6.0.jar
```

```python
import json

zookeeper_host = 'host1'
hbase_table_name = 'testTable'

conf = {"hbase.zookeeper.quorum": zookeeper_host, "hbase.mapreduce.inputtable": hbase_table_name}
keyConv = "org.apache.spark.examples.pythonconverters.ImmutableBytesWritableToStringConverter"

# 注意这里，使用自己定义的Converter读取
valueConv = "org.apache.spark.examples.pythonconverters.MyHBaseResultToStringConverter"

hbase_rdd = sc.newAPIHadoopRDD(
        "org.apache.hadoop.hbase.mapreduce.TableInputFormat",
        "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
        "org.apache.hadoop.hbase.client.Result",
        keyConverter=keyConv,
        valueConverter=valueConv,
        conf=conf)
hbase_rdd = hbase_rdd.flatMapValues(lambda v: v.split("\n")).mapValues(json.loads)

hbase_rdd.take(1)
```

然后就可以看到结果了。

以上就是如何通过修改 HBaseConverters.scala 让 pyspark 从 HBase 中读取 java 的特殊类型。