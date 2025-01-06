---
categories:
- 分布式平台
date: 2019-04-10 17:26:30+0000
description: 应甲方需求，写一个 pyspark 读写 HBase 的教程。主要包含了基本读写方法和自定义 Converter 的方法。
draft: false
math: null
tags:
- Spark
- software
title: pyspark读写HBase
---
应甲方需求，写一个 pyspark 读写 HBase 的教程。主要包含了基本读写方法和自定义 Converter 的方法。
<!--more-->

# pyspark 读取 HBase

以下内容的环境：python 3.5，spark 1.6

pyspark 读取 HBase 需要借助 Java 的类完成读写。

首先需要明确的是，HBase 中存储的是 `byte[]`，也就是说，不管是什么样的数据，都需要先转换为 `byte[]` 后，才能存入 HBase。

## 基本方法
pyspark 读取 HBase 需要使用 `SparkContext` 的 [newAPIHadoopRDD](http://spark.apache.org/docs/1.6.0/api/python/pyspark.html#pyspark.SparkContext.newAPIHadoopRDD) 这个方法，这个方法需要使用 Java 的类，用这些类读取 HBase

下面的示例代码默认 HBase 中的行键、列族名、列名和值都是字符串转成的 `byte` 数组：

read_hbase_pyspark.py
```python
# -*- coding:utf-8 -*-
import json

from pyspark import SparkContext
from pyspark import SparkConf

if __name__ == "__main__":
    conf = SparkConf().set("spark.executorEnv.PYTHONHASHSEED", "0")\
                      .set("spark.kryoserializer.buffer.max", "2040mb")
    sc = SparkContext(appName='HBaseInputFormat', conf=conf)

    # 配置项要包含 zookeeper 的 ip
    zookeeper_host = 'zkServer'

    # 还要包含要读取的 HBase 表名
    hbase_table_name = 'testTable'    

    conf = {"hbase.zookeeper.quorum": zookeeper_host, "hbase.mapreduce.inputtable": hbase_table_name}

    # 这个Java类用来将 HBase 的行键转换为字符串
    keyConv = "org.apache.spark.examples.pythonconverters.ImmutableBytesWritableToStringConverter"
    # 这个Java类用来将 HBase 查询得到的结果，转换为字符串
    valueConv = "org.apache.spark.examples.pythonconverters.HBaseResultToStringConverter"

    # 第一个参数是 hadoop 文件的输入类型
    # 第二个参数是 HBase rowkey 的类型
    # 第三个参数是 HBase 值的类型
    # 这三个参数不用改变
    # 读取后的 rdd，每个元素是一个键值对，(key, value)
    hbase_rdd = sc.newAPIHadoopRDD(
        "org.apache.hadoop.hbase.mapreduce.TableInputFormat",
        "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
        "org.apache.hadoop.hbase.client.Result",
        keyConverter=keyConv,
        valueConverter=valueConv,
        conf=conf)

    # 读取后，将键值对 (key, value) 中的值 value，使用\n切分，用 flatMap 展开
    # 然后将键值对 (key, value) 中的值 value 使用 json.loads 解析，得到 dict
    hbase_rdd = hbase_rdd.flatMapValues(lambda v: v.split("\n")).mapValues(json.loads)

    output = hbase_rdd.collect()
    for (k, v) in output:
        print((k, v))
```

上述代码在提交给 spark 集群的时候，要指名用到的 Java 类的位置，这些类都在 spark-examples 这个包里面，这个包在 spark 目录下的 lib 里面。以 CDH 5.7.2 为例，CDH 集群中这个包的位置在 `/opt/cloudera/parcels/CDH-5.7.2-1.cdh5.7.2.p0.18/lib/spark/lib/spark-examples-1.6.0-cdh5.7.2-hadoop2.6.0-cdh5.7.2.jar`，所以提交命令为：

```bash
spark-submit --master yarn --jars /opt/cloudera/parcels/CDH-5.7.2-1.cdh5.7.2.p0.18/lib/spark/lib/spark-examples-1.6.0-cdh5.7.2-hadoop2.6.0-cdh5.7.2.jar read_hbase_pyspark.py
```

所以，上述的 Java 类，核心都是认为 HBase 中所有的值，原本都是字符串，然后转换成 `byte` 数组后存入的 HBase，它在解析的时候，将读取到的 `byte[]` 转换为字符串后返回，所以我们拿到的值就是字符串。

## 进阶方法

对于其他类型的数据，转换为 `byte` 数组后存入 HBase，如果我们还使用上面的 Java 类去读取 HBase，那么我们拿到的字符串的值就是不正确的。

为了理解这些内容，我们首先要讨论 HBase 中值的存储结构。

HBase 是非结构化数据库，以行为单位，每行拥有一个行键 rowkey，对应的值可以表示为一个 map（python 中的 dict），举个例子，如果我们有一条记录，行键记为 "r1"，里面有 1 个列族(columnFamily) "A"，列族中有两列(qualifier)，分别记为 "a" 和 "b"，对应的值分别为 "v1" 和 "v2"，那么表示成 json 字符串就是下面的形式：

```python
{
    "r1": {
        "A" : {
            "a": "v1",
            "b": "v2"
        }
    }
}
```

上面这个 json 字符串就是上面那条记录在 HBase 中存储的示例，第一层的键表示行键(rowkey)，对应的值表示这一行的值；第二层的键表示列族名(columnFamily)，值表示这个列族下列的值；第三层的键表示列名(qualifier)，对应的值(value)表示这个由行键、列族名、列名三项确定的一个单元格(Cell)内的值。所以上面这个例子中，只有一行，两个单元格。

下面我们针对 pyspark 读取 HBase 使用到的 `org.apache.spark.examples.pythonconverters.HBaseResultToStringConverter` 来讨论。

Java 的 API 在读取 HBase 的时候，会得到一个 `Result` 类型，这个 `Result` 就是查询结果。`Result` 可以遍历，里面拥有多个 `Cell`，也就是单元格。上面我们说了，每个单元格至少有 4 个内容：行键、列族名、列名、值。

`HBaseResultToStringConverter` 是由 scala 实现的一个类，它的功能是将 Java HBase API 的 `Result` 转换为 `String`，源码如下：

```scala
package org.apache.spark.examples.pythonconverters

import scala.collection.JavaConverters._
import scala.util.parsing.json.JSONObject

import org.apache.spark.api.python.Converter
import org.apache.hadoop.hbase.client.{Put, Result}
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.hbase.KeyValue.Type
import org.apache.hadoop.hbase.CellUtil

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

它完成的工作是遍历 `Result` 中的 `Cell`，每个 `Cell` 转换成一个 scala `Map`，键分别是行键、列族名、列名、时间戳、HBase 操作类型、值。最后每个 scala `Map` 被转换成 json 字符串，之间用 '\n' 分隔。

这里的 `CellUtil.CloneRow`，`CellUtil.cloneFamily`，`CellUtil.cloneQualifier`，`CellUtil.cloneValue` 是我们主要使用的四个方法，这四个方法生成的都是 `byte[]`，然后这四个 `byte[]` 都被 `Bytes.toStringBinary` 转换成了 `String` 类型。

所以，如果我们存入 HBase 的数据是 `String` 以外类型的，如 `Float`, `Double`, `BigDecimal`，那么这里使用 `CellUtil` 的方法拿到 `byte[]` 后，需要使用 `Bytes` 里面的对应方法转换为原来的类型，再转成字符串或其他类型，生成 json 字符串，然后返回，这样我们通过 pyspark 才能拿到正确的值。

下面是一个示例，我们的数据都是 `java.math.BigDecimal` 类型的值，存 HBase 的时候将他们转换为 `byte[]` 后进行了存储。那么解析的时候，就需要自定义一个处理 `BigDecimal` 的类：`HBaseResultToBigDecimalToStringConverter`

```scala
package org.apache.spark.examples.pythonconverters

import java.math.BigDecimal

import scala.collection.JavaConverters._
import scala.util.parsing.json.JSONObject

import org.apache.spark.api.python.Converter
import org.apache.hadoop.hbase.client.{Put, Result}
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.hbase.KeyValue.Type
import org.apache.hadoop.hbase.CellUtil

class HBaseResultToBigDecimalToStringConverter extends Converter[Any, String] {
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

上述代码中，引入了 `java.math.BigDecimal`，将 `value` 的解析进行了简单的修改，通过 `CellUtil.cloneValue` 拿到 `byte[]` 后，通过 `Bytes.toBigDecimal` 转换成 `java.math.BigDecimal`，然后使用 `toString` 方法转换成字符串。

这个类写完后，我们就可以对其进行编译，导出成 jar 包，在 pyspark 程序中指明，读取的时候，使用这个类解析 value。

这样源代码就改完了，需要编译成 jar 包。

首先安装 [maven](http://maven.apache.org/) 3.6.0，下载后，解压，配置环境变量即可。

下载 spark 的源码，去 Apache Spark 官网，下载仓库中的源代码 [spark-1.6.0.tgz](https://archive.apache.org/dist/spark/spark-1.6.0/) 。

下载后解压，将根目录中的 scalastyle-config.xml 拷贝到 examples 目录下。

修改 `examples/src/main/scala/org/apache/spark/examples/pythonconverters/HBaseConverters.scala`，增加自己用的类。

修改 `examples/pom.xml`，将 `<artifactId>spark-examples_2.10</artifactId>` 修改为 `<artifactId>spark-examples_2.10_my_converters</artifactId>`。

cd 到 examples 目录下，使用以下命令编译 spark-examples

```
mvn clean install -pl :spark-examples_2.10_my_converters
```

编译途中保证全程联网，编译的时候会有一些警告，编译好的包在同级目录下的 target 中，有个叫 spark-examples_2.10_my_converters-1.6.0.jar 的文件。

然后就是使用这个包读取 HBase 中的 BigDecimal了：

我们使用 standalone 模式运行 pyspark 交互式界面：

```bash
pyspark --master spark://host1:7077 --jars spark-examples_2.10_my_converters-1.6.0.jar
```

执行以下内容：

```python
import json

zookeeper_host = 'host1'
hbase_table_name = 'testTable'

conf = {"hbase.zookeeper.quorum": zookeeper_host, "hbase.mapreduce.inputtable": hbase_table_name}
keyConv = "org.apache.spark.examples.pythonconverters.ImmutableBytesWritableToStringConverter"

# 注意这里，使用自己定义的Converter读取
valueConv = "org.apache.spark.examples.pythonconverters.HBaseResultToBigDecimalToStringConverter"

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

然后就可以看到结果了，如何验证读取的对不对呢，可以尝试将 `valueConv` 改回 `HBaseResultToStringConverter`，然后观察 value 的值。

以上就是如何通过修改 HBaseConverters.scala 让 pyspark 从 HBase 中读取 `java.math.BigDecimal` 的示例。

# pyspark 写入 HBase

pyspark 写入 HBase 使用 `SparkContext` 的 [saveAsNewAPIHadoopDataset](http://spark.apache.org/docs/1.6.0/api/python/pyspark.html#pyspark.RDD.saveAsNewAPIHadoopDataset)，和读取的方法类似，也需要使用 Java 的类。

**下面的方法要求存入 HBase 中的数据，行键、列族名、列名、值都为字符串**

write_into_hbase_pyspark.py
```python
# -*- coding:utf-8 -*-
from pyspark import SparkContext
from pyspark import SparkConf

if __name__ == "__main__":
    conf = SparkConf().set("spark.executorEnv.PYTHONHASHSEED", "0")\
                      .set("spark.kryoserializer.buffer.max", "2040mb")
    sc = SparkContext(appName='HBaseOutputFormat', conf=conf)

    # 配置项要包含 zookeeper 的 ip
    zookeeper_host = 'zkServer'

    # 还要包含要写入的 HBase 表名
    hbase_table_name = 'testTable'    

    conf = {"hbase.zookeeper.quorum": zookeeper_host,
            "hbase.mapred.outputtable": hbase_table_name,
            "mapreduce.outputformat.class": "org.apache.hadoop.hbase.mapreduce.TableOutputFormat",
            "mapreduce.job.output.key.class": "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
            "mapreduce.job.output.value.class": "org.apache.hadoop.io.Writable"}
    keyConv = "org.apache.spark.examples.pythonconverters.StringToImmutableBytesWritableConverter"
    valueConv = "org.apache.spark.examples.pythonconverters.StringListToPutConverter"

    records = [
        ['row1', 'f1', 'q1', 'value1'],
        ['row2', 'f1', 'q1', 'value2'],
        ['row3', 'f1', 'q1', 'value3'],
        ['row4', 'f1', 'q1', 'value4']
    ]

    sc.parallelize(records)\
      .map(lambda x: (x[0], x))\
      .saveAsNewAPIHadoopDataset(
        conf=conf,
        keyConverter=keyConv,
        valueConverter=valueConv)
```

首先在控制台启动 HBase-shell

```bash
hbase shell
```

然后创建表，表名为 testTable，只有一个列族，列族名为 f1：

```bash
create 'testTable', 'f1'
```

使用 `quit` 退出 HBase-shell

提交 pyspark 程序：

```
spark-submit --master spark://master:7077 --jars /opt/cloudera/parcels/CDH-5.7.2-1.cdh5.7.2.p0.18/lib/spark/lib/spark-examples-1.6.0-cdh5.7.2-hadoop2.6.0-cdh5.7.2.jar write_into_hbase_pyspark.py
```

运行完成后，再次进入 HBase-shell，运行：

```bash
scan 'testTable'
```

可以看到类似下面的输出结果：

```
hbase(main):001:0> scan 'testTable'
ROW                           COLUMN+CELL
 row1                         column=f1:q1, timestamp=1554892784494, value=value1
 row2                         column=f1:q1, timestamp=1554892784494, value=value2
 row3                         column=f1:q1, timestamp=1554892816961, value=value3
 row4                         column=f1:q1, timestamp=1554892816961, value=value4
4 row(s) in 0.3330 seconds
```

这就完成了写入 HBase 的过程。

**需要注意的是：rdd 中的每个元素，都必须是一个列表(`list`)，不能是其他类型，如 `tuple`，而且每个列表内必须是 4 个元素，分别表示 `[行键、列族名、列名、值]`，且每个元素都为 `str` 类型。**

原因是 `StringListToPutConverter` 这个类做转换的时候需要将 rdd 中的元素，看作是一个 `java.util.ArrayList[String]`

```scala
class StringListToPutConverter extends Converter[Any, Put] {
  override def convert(obj: Any): Put = {
    val output = obj.asInstanceOf[java.util.ArrayList[String]].asScala.map(Bytes.toBytes).toArray
    val put = new Put(output(0))
    put.add(output(1), output(2), output(3))
  }
}
```

`StringListToPutConverter` 的工作原理是，将传入的元素强制类型转换为 `java.util.ArrayList[String]`，将第一个元素作为行键、第二个元素作为列族名、第三个元素作为列名、第四个元素作为值，四个值都转换为 `byte[]` 后上传至 HBase。

所以我们可以修改这个类，实现存入类型的多样化。

举个例子，如果我想存入一个 `java.math.BigDecimal`，那实现的方法就是：在 pyspark 程序中，将数字转换成 `str` 类型，调用我们自己写的一个 converter：

```scala
import java.math.BigDecimal

class StringListToBigDecimalToPutConverter extends Converter[Any, Put] {
  override def convert(obj: Any): Put = {
    val output = obj.asInstanceOf[java.util.ArrayList[String]].asScala.toArray
    val put = new Put(Bytes.toBytes(output(0)))
    put.add(
        Bytes.toBytes(output(1)),
        Bytes.toBytes(output(2)),
        Bytes.toBytes(new BigDecimal(output(3)))
    )
  }
}
```

就可以实现存入的值是 `java.math.BigDecimal` 了。

# CDH 5.9 以前的版本，python3，master 选定为 yarn 时的 bug

CDH 5.9 以前的版本在使用 yarn 作为 spark master 的时候，如果使用 python3，会出现 yarn 內部 `topology.py` 这个文件引发的 bug。这个文件是 python2 的语法，我们使用 python3 运行任务的时候，python3 的解释器在处理这个文件时会出错。

解决方案是：将这个文件重写为 python3 的版本，每次在重启 yarn 之后，将这个文件复制到所有机器的 `/etc/hadoop/conf.cloudera.yarn/`目录下。

以下是 python3 版本的 `topology.py`。

`topology.py`
```python
#!/usr/bin/env python
#
# Copyright (c) 2010-2012 Cloudera, Inc. All rights reserved.
#

'''
This script is provided by CMF for hadoop to determine network/rack topology.
It is automatically generated and could be replaced at any time. Any changes
made to it will be lost when this happens.
'''

import os
import sys
import xml.dom.minidom

def main():
  MAP_FILE = '{{CMF_CONF_DIR}}/topology.map'
  DEFAULT_RACK = '/default'

  if 'CMF_CONF_DIR' in MAP_FILE:
    # variable was not substituted. Use this file's dir
    MAP_FILE = os.path.join(os.path.dirname(__file__), "topology.map")

  # We try to keep the default rack to have the same
  # number of elements as the other hosts available.
  # There are bugs in some versions of Hadoop which
  # make the system error out.
  max_elements = 1

  map = dict()

  try:
    mapFile = open(MAP_FILE, 'r')

    dom = xml.dom.minidom.parse(mapFile)
    for node in dom.getElementsByTagName("node"):
      rack = node.getAttribute("rack")
      max_elements = max(max_elements, rack.count("/"))
      map[node.getAttribute("name")] = node.getAttribute("rack")
  except:
    default_rack = "".join([ DEFAULT_RACK for _ in range(max_elements)])
    print(default_rack)
    return -1

  default_rack = "".join([ DEFAULT_RACK for _ in range(max_elements)])
  if len(sys.argv)==1:
    print(default_rack)
  else:
    print(" ".join([map.get(i, default_rack) for i in sys.argv[1:]]))
  return 0

if __name__ == "__main__":
  sys.exit(main())
```