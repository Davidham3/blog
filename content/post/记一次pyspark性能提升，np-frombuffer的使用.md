---
categories:
- 分布式平台
date: 2020-02-13 00:12:07+0000
draft: false
math: true
tags:
- Spark
- software
title: 记一次pyspark性能提升，np.frombuffer的使用
---
之前项目中有个任务是读取一堆二进制文件，对二进制文件进行解析，然后存到HBase。由于有 .mat 文件，整个 spark 都用 pyspark 写来着，也没用 scala。最近天天都在写文档啥的，还得写毕业论文，觉得太没劲了就研究了一下优化的问题，顺便更新下博客，好久没更新了。
<!--more-->

原来的数据格式很简单，就是一堆 float 类型的数字，转成了二进制的形式，每 4 个字节一个数，连续地写到了文件里面。因为文件很多，而且要存到 HBase，就选择用 Spark 来处理。每个文件差不多 300M，里面的数字很多，最后的目标是提取成文件内数据为矩阵的形式。每个文件里面有 $K \times M \times N$ 个浮点型数字。要提取成 $K$ 个 $M \times N$ 的矩阵。

假设 $K = 500$，$M = 5000$, $N = 30$，首先写一个数据生成器出来：

data_generator.py
```python
# -*- coding:utf-8 -*-

import numpy as np

K = 500
M = 5000
N = 30
c = K * M * N
np.random.uniform(size=(c,)).astype(np.float32).tofile('test.data')

```

原来不知道有这个 np.tofile，今天才知道的。。。速度很快，准确的说是太快了。。。

数据使用 sparkContext 的 binaryFiles 就可以读到内存中，以 bytes 的形式存储。

原来我提取 $K$ 个矩阵的方法是：

```python
K = 500
M = 5000
N = 30

UNIT = M * N
UNIT_ = UNIT * 4


def record_time(f):
    def wrapper(content):
        t = time.time()
        f(content)
        print(time.time() - t)
    return wrapper


@record_time
def method1(content):
    for i in range(K):
        np.array(
            struct.unpack(
                '{}f'.format(UNIT),
                content[i * UNIT_: (i + 1) * UNIT_]
            )
        ).reshape(M, N)
```

直接用 struct.unpack $K$ 次，而且每次都用 np.array 构造出新的 array。但是不得不说，因为 python 底层是 C 写的，这个 struct.unpack 很快，完成上面的解析只需要 10s。当时想着 python 解析这个的速度肯定没有 scala 快，写成这样应该就差不多了。但是最近我都没怎么写程序，就想研究研究能不能优化。

## 优化方案1：使用 C++ 重写这部分

我当时觉得，使用 C++ 完成 bytes 到 np.ndarray 的转换应该快一点，然后就写了一个：

tools.cpp
```cpp
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <cstring>
#include <time.h>
#include <Eigen/Dense>
#include<pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace std;
using Eigen::MatrixXd;

MatrixXd extract_matrix(long long k_idx, char * stream, int M, int N) {
    float data;
    
    MatrixXd m(M, N);
    long offset = k_idx * (M * N);
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++) {
            memcpy(&data, stream + offset, 4);
            m(r, c) = data;
            offset += 4;
        }
    return m;
}

PYBIND11_MODULE(tools, m) {
    m.def("extract_matrix", &extract_matrix, pybind11::return_value_policy::reference);
}
```

实际上是利用 pybind11 实现了一个 C++ 函数，使用下面的命令编译：

```
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -I /mnt/c/Users/chaosong/Downloads/eigen-3.3.7 tools.cpp -o tools`python3-config --extension-suffix`
```

然后 python 就可以导入了

```python
import tools


@record_time
def method2(content):
    for i in range(K):
        tools.extract_matrix(i, content, M, N).reshape(M, N)
```

结果这玩意出奇的慢。。。用这个 method2 处理那个文件要 254s。比原来的方法慢了 25 倍。

## 优化方案2：使用 np.frombuffer

我去看了看 numpy 的文档，结果发现了这个函数，这个函数可以直接读 bytes，都不需要用 struct 解析了。

```python
@record_time
def method3(content):
    for i in range(K):
        np.frombuffer(
            content[i * UNIT_: (i + 1) * UNIT_],
            dtype=np.float32,
            count=UNIT
        ).reshape(M, N)
```

这里我故意写成切片的形式，为了和优化方案3的速度做对比。但即便是循环 K 次，将原始数据用切片进行了复制，这个方法的时间都能达到 0.03s，比我那个失败的方案 1 快了 8467 倍，比原方案快了 333 倍。

## 优化方案3：使用 np.frombuffer

```python
@record_time
def method4(content):
    for i in range(K):
        np.frombuffer(
            content,
            dtype=np.float32,
            count=UNIT,
            offset=i * UNIT_
        ).reshape(M, N)
```

这个能跑到 0.0007s。比方案 1 快 36 万倍，比原方案快 14285 倍，比方案 2 快 42 倍。

结论：
1. 我的那个方案 1 太差劲了，肯定是我不会写才导致的。。。看看以后啥时候有时间研究一下
2. np.frombuffer，太强了。。。
3. 多看 API。
4. python 的 struct 其实挺快的，但是基于 C/C++ 的 numpy 更快，无敌快。

附上完整代码：

```python
# -*- coding:utf-8 -*-

import time
import struct
import numpy as np
import tools

with open('test.data', 'rb') as f:
    content = f.read()

K = 500
M = 5000
N = 30

UNIT = M * N
UNIT_ = UNIT * 4


def record_time(f):
    def wrapper(content):
        t = time.time()
        f(content)
        print(time.time() - t)
    return wrapper


@record_time
def method1(content):
    for i in range(K):
        np.array(
            struct.unpack(
                '{}f'.format(UNIT),
                content[i * UNIT_: (i + 1) * UNIT_]
            )
        ).reshape(M, N)


@record_time
def method2(content):
    for i in range(K):
        tools.extract_matrix(i, content, M, N).reshape(M, N)


@record_time
def method3(content):
    for i in range(K):
        np.frombuffer(
            content[i * UNIT_: (i + 1) * UNIT_],
            dtype=np.float32,
            count=UNIT
        ).reshape(M, N)


@record_time
def method4(content):
    for i in range(K):
        np.frombuffer(
            content,
            dtype=np.float32,
            count=UNIT,
            offset=i * UNIT_
        ).reshape(M, N)


if __name__ == "__main__":
    for f in [method1, method2, method3, method4]:
        f(content)

```

接下来需要验证的是，我们的 pyspark 程序比原来快了多少，这个明天再验证吧。