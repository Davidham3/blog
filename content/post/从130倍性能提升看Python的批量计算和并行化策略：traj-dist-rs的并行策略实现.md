---
date: '2026-02-24'
draft: false
tags:
- trajectory
- rust
- parallel
math: true
title: '从130倍性能提升看Python的批量计算和并行化策略：traj-dist-rs的并行策略实现'
---

在上一篇文章中，我分享了如何用 Rust 重写 `traj-dist`，实现了单次距离计算的性能提升。但是，当面对大规模轨迹数据集时，单次调用并不是最优雅的解决方案。

举个例子，如果我们要做轨迹序列的聚类，那么就一定要计算任意两条轨迹之间的距离。以动态规划类算法为例，时间复杂度是`O(NM)`。假如有 1000 条轨迹，需要计算它们之间的所有距离对。那就是 $1000 \times 1000 \times MN$，假设轨迹平均长度是1000，那至少有1000^4的复杂度。那如果要算10万条轨迹之间的距离呢？1000万呢？这将是非常大的计算量（肯定要上分布式了）。

在这篇文章中，我会分享 `traj-dist-rs` 的批量计算接口和并行化策略，以及如何通过正确的技术路线实现超过**130倍**的性能提升。

<!--more-->

对于轨迹序列聚类、knn等算法，需要计算所有轨迹之间的距离。

问题定义：

给定K条轨迹（`List[np.ndarray]`），计算它们两两之间的距离，最终得到一个`(K, K)`的距离矩阵。为避免冗余计算，我们只计算矩阵的上三角部分。

我们还是以`traj-dist`作为我们实验的基线。

## 一、基准测试：传统方案的性能瓶颈

给定20条轨迹，每条轨迹的长度为1000，计算距离矩阵的上三角部分，距离使用DTW。因此一共需要计算190次（对角线不计算）。

解决这个问题，有3种简单的解决思路：

1. 使用双重for循环。
2. 使用`traj-dist`提供的pdist函数。
3. 使用双重for循环+joblib。

示例代码如下：

```python
import time
import numpy as np
from joblib import Parallel, delayed
from tqdm import trange
import traj_dist.distance as tdist

NUM_TRAJECTORIES = 20
TRAJS = [np.random.uniform(size=(1000, 2)) for _ in range(NUM_TRAJECTORIES)]
WARMUP_RUNS = 1


def test_func(func, num_runs):
    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = func()

    # Measure time
    times = []
    for _ in trange(num_runs):
        start = time.perf_counter()
        _ = func()
        end = time.perf_counter()
        times.append(end - start)
    return np.median(times)


if __name__ == "__main__":
    def test1():
        for i in range(NUM_TRAJECTORIES):
            for j in range(i + 1, NUM_TRAJECTORIES):
                tdist.dtw(TRAJS[i], TRAJS[j], "euclidean")

    def test2():
        tdist.pdist(TRAJS, metric="dtw", type_d="euclidean")

    def test3():
        Parallel(n_jobs=-1)(
            delayed(lambda x, y: tdist.dtw(x, y, "euclidean"))(TRAJS[i], TRAJS[j])
            for i in range(NUM_TRAJECTORIES)
            for j in range(i + 1, NUM_TRAJECTORIES)
        )

    print(test_func(test1, 5))
    print(test_func(test2, 5))
    print(test_func(test3, 5))
```

跑出来的耗时如下：

|tool     | technique                    | running time(s) | speedup |
|---------|------------------------------|-----------------|---------|
|traj-dist| **Route 1**: double for-loop | 10.103s         | 1x      |
|traj-dist| **Route 2**: traj-dist pdist | 10.088s         | 1.001x  |
|traj-dist| **Route 3**: joblib parallel | 1.364s          | 7.407x  |

`traj-dist`的结果还是有点让人惊讶的，因为pdist相较于双重for循环几乎没有任何提升，说明pdist没有消除python解释器带来的不利。赶紧去看了一下源码，才发现pdist居然是在python里面通过双重for循环实现的，没有用cython加速😂。那我们后续都不用考虑这个函数了。

joblib并行有比较明显的提升，因为joblib默认使用loky作为后端启动多进程，多个进程同时计算，我测试用的机器是20个CPU核心，joblib在设置`n_jobs=-1`的时候，会启动和CPU相同数量的进程数。

## 二、第一层加速：零拷贝与Rust的降维打击

我们使用`traj-dist-rs`完成相同的实验，看看效果：

|tool        | technique                    | running time(s) | speedup |
|------------|------------------------------|-----------------|---------|
|traj-dist-rs| **Route 1**: double for-loop | 0.631s          | 16.011x |
|traj-dist-rs| **Route 3**: joblib parallel | 0.105s          | 96.219x |

可以看到，双重for循环的方案比`traj-dist`快了16.011倍，如果使用Joblib，达到了96.219倍。

这个结果还是比较惊人的，在上一篇文章里面我们讲了，`traj-dist-rs`的核心提升在于引入零拷贝，让rust直接读取python中numpy.ndarray的底层数据。

这说明当前`traj-dist-rs`使用的零拷贝设计配合rust的高性能可以轻松超越`traj-dist`使用cython编写的加速代码。

但是，`traj-dist-rs`还没有实现`pdist`函数，考虑到python用户为了加速一定会使用多进程加速，那么`traj-dist-rs`的`pdist`函数在设计上就一定要考虑到并行能力。

## 三、并行方案构想

我们分别讨论python的多进程并行与rust的rayon并行。

### 3.1 Python多进程并行

众所周知，GIL限制了Python的并行效率，只能通过多进程实现并行加速计算，可以使用python自带的多进程库，如multiprocessing，也可以使用类似joblib这样的库。本质上都是通过多进程实现并行。

然而，不论是哪种并行，一定会遇到下面的3个问题：

#### 1. 初始化开销

本质是python主进程会启动多个子进程，这些子进程的启动是有一定的开销的。当计算量变大的时候，这个开销就可以忽略，但是计算量小的时候，进程的初始化反而会成为瓶颈。

#### 2. 通信开销

进程之间的数据是隔离的，不能共享，使用多进程的时候，大概会经历下面的步骤：

1. 参数和函数会以pickle的形式序列化，由主进程传递到另一个python进程（子进程）；
2. 子进程反序列化，进行运算，运算结果通过pickle序列化传递到主进程；
3. 主进程反序列化拿到最终结果，收集所有子进程的结果。

可以看到一共是2次信息传递、2次序列化、2次反序列化。数据量越大，这个开销越大，当然也是有避开的方案的，比如使用共享内存：提前将数据写入共享内存，子进程从共享内存读取这部分数据。本文先不讨论这个方案。

#### 3. 调度与负载均衡开销

主要有2点：

1. 负载不均：如果1个进程执行的都是一些计算量比较大的工作，其他进程都是轻量的工作，那么其他进程完成任务后就会闲置，就像木桶原理一样。
2. 任务调度开销：为了解决上面的问题，可以用一个队列维护任务，每个进程从队列里面拉取任务进行消费，但是单个任务的任务量需要设为多大？如果设置的小了，子进程会频繁拉取，每次拉取都有固定开销，造成总开销变大；设置的大了对内存压力又比较大，而且又有可能造成负载不均的问题。

但是说了这么多，Python里面为了加速运算，多进程并行一定是避不开的一个方案，我个人还是喜欢共享内存的方案，尤其是结合pyarrow。

回到上面的实验，可以看到joblib确实可以显著提升性能，因此对于大部分场景来说，使用joblib只需要几行代码就可以快速提升性能，这是一个很不错的方案，相当于1行代码换取成倍提升。

不过这里也需要多讲一句，joblib的一大优势是为python用户提供了非常Pythonic的并行接口，通过简单的`Parallel(n_jobs=-1)(delayed(...))`实现多进程代码实现，这是非常优雅的。

### 3.2 Rust+Rayon的并行加速方案

rust里面做并行，肯定避不开使用Rayon，Rayon已经成为rust生态中数据并行的事实标准。

Rayon自身的优势有几点：

1. 极致的易用性：像joblib一样提供了非常傻瓜的使用方式，改造成本极低。
2. 无畏并发：编译时就可以保证安全，大部分情况无需加锁。
3. 工作窃取：某个线程完成自己所有任务之后会去其他繁忙线程中窃取任务。

相比上面的python多进程：

1. 初始化开销：rayon用线程，开销更低。
2. 通信开销：数据跨线程共享。
3. 调度与负载均衡开销：工作窃取。

因此，`traj-dist-rs`的`pdist`一定会使用Rayon完成并行计算的工作，以实现最佳性能。

## 四、巅峰对决：130倍性能提升

我们直接看通过rayon加速后的pdist的性能吧，后面再看具体实现。

我们在第一节的实验里面给出了3种路线：

1. 使用双重for循环。
2. 使用`traj-dist`提供的pdist函数。（实际上这个和第一个一样）
3. 使用双重for循环+joblib。

`traj-dist-rs`提供的pdist函数支持了串行和并行的选项可以选择，因此就形成了4条路线：

1. 使用双重for循环。
2. 使用`traj-dist-rs`提供的pdist函数（串行）。
3. 使用双重for循环+joblib。
4. 使用`traj-dist-rs`提供的pdist函数（并行）。

这里讲一下上面的4条技术路线：

| 技术路线 | 特点 |
| --------- | ------ |
| Route1: 双重for循环 | 这个方案调用的dtw是traj-dist-rs优化过的，t1与t2从python转移到rust的时候是零拷贝，性能很好。但是双重for循环在python里面会比较慢，因为cpython的解释操作会拖累这里的性能。 |
| Route2: rust串行 | rust可以通过零拷贝直接读取TRAJS里面的数据，内部也是通过双重for循环调用dtw函数进行计算，但是这个for循环会比cpython快多了。 |
| Route3: joblib并行 | 通过joblib实现多进程并行（默认是loky后端），虽然调用的是traj-dist-rs的dtw，但是t1和t2要从主进程转移到子进程，这里会发生数据序列化与反序列化，就相当于输入数据复制了两次；而返回值也是要做一次序列化和反序列化，因此也是两次。但是对于python来说，是个不错的并行方案。 |
| Route4: rayon并行 | rust通过零拷贝读取TRAJS里面的数据，rayon通过多线程执行dtw，与上面的joblib类似，不过没有跨进程数据传输、进程维护开销。 |

下面有两个测试结果，仍然取20条轨迹，每次都是取5次测量的中位数。

### 4.1 高计算负载：轨迹长度=1000

|tool        | technique                           | running time(s) | speedup  |
|------------|-------------------------------------|-----------------|----------|
|traj-dist   | **Route 1**: double for-loop        | 10.103s         | 1x       |
|traj-dist   | **Route 3**: joblib parallel        | 1.364s          | 7.407x   |
|traj-dist-rs| **Route 1**: double for-loop        | 0.631s          | 16.011x  |
|traj-dist-rs| **Route 2**: rust serial (pdist)    | 0.628s          | 16.088x  |
|traj-dist-rs| **Route 3**: joblib parallel        | 0.105s          | 96.219x  |
|traj-dist-rs| **Route 4**: rayon parallel (pdist) | 0.078s          | 129.526x |

结果分析：

1. Python循环开销 vs Rust循环开销：`traj-dist-rs`的Python循环（0.631s）和Rust串行pdist（0.628s）耗时几乎相同。这说明在高计算负载下，DTW算法本身的耗时占据主导，Python循环的开销显得不那么重要。
2. 多进程 vs 多线程：Rayon并行（0.078s）明显优于Joblib（0.105s），这得益于其更低的开销和更高效的线程间协作。
3. 最终的胜利：`traj-dist-rs`的原生并行 pdist 接口，相较于最初的`traj-dist`基准，实现了近**130倍**的性能飞跃！这正是我们追求的更高性能。

### 4.2 低计算负载：轨迹长度=10

|tool        | technique                           | running time(s) | speedup  |
|------------|-------------------------------------|-----------------|----------|
|traj-dist   | **Route 1**: double for-loop        | 0.00186s        | 1x       |
|traj-dist   | **Route 3**: joblib parallel        | 0.0653s         | 0.028x   |
|traj-dist-rs| **Route 1**: double for-loop        | 0.000139s       | 13.381x  |
|traj-dist-rs| **Route 2**: rust serial (pdist)    | 0.0000711s      | 26.160x  |
|traj-dist-rs| **Route 3**: joblib parallel        | 0.0545s         | 0.034x   |
|traj-dist-rs| **Route 4**: rayon parallel (pdist) | 0.00201s        | 0.925x   |

结果分析：

1. 并行化的代价：所有并行方案（Joblib和Rayon）的性能都不如串行。Joblib的进程启动和数据序列化开销尤为巨大，导致性能下降了两个数量级。Rayon虽然开销小得多，但依然不敌最快的串行实现。
2. 解释器开销的凸显：在低计算负载下，`traj-dist-rs`的Rust串行pdist比其Python循环快了近2倍。这说明当核心计算非常快时，Python解释器本身那微不足道的循环开销就成了主要瓶颈。
3. 最佳策略：在这种场景下，最快的方案是调用`traj-dist-rs`的串行 pdist 函数。它既避免了Python的循环开销，也避免了并行的管理开销。

## 五、具体实现：traj-dist-rs的性能魔法

说了这么多，核心还是要讲`traj-dist-rs`为了批量运算，做了哪些工作。

### 5.1 Rayon：一行代码解锁并行

最核心的地方就在于rayon。因为`traj-dist-rs`已经支持了零拷贝读取python中的numpy.ndarray，那么只需要很简单的使用rayon做并行计算就好了。

```rust
fn compute_pdist_parallel<T, D>(trajectories: &[T], calculator: &D) -> Vec<f64>
where
    T: CoordSequence + Sync,
    D: Distance<T>,
{
    let n = trajectories.len();

    // Create index pairs for all unique pairs (i, j) where i < j
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    // Compute distances in parallel using Rayon's global thread pool
    pairs
        .into_par_iter()
        .map(|(i, j)| calculator.distance(&trajectories[i], &trajectories[j]))
        .collect()
}
```

就这么简单，核心就是引入`into_par_iter`，然后就自动并行了。这里的`Distance`提供了一个叫distance的方法，用于计算两条轨迹之间的距离，比如dtw，edr等算法。

也就是说，我们从python里面传递给rust的是一个`List[np.ndarray]`，rust拿到之后将其转换为`&[CoordSequence]`对应的具体类型，就可以复用这个并行计算的pdist函数了。至于`np.ndarray`如何转`CoordSequence`，这个就是上次讲的，通过PyO3取出`ndarray`底层的切片，然后封装到一个实现了`CoordSequence`这个trait的struct里面就可以了。那`List`如何转`&[]`，这个就不用说了。

### 5.2 Bincode：为Pickle序列化加速

因为`traj-dist-rs`支持python，所以一定要考虑类似上面通过multiprocessing或者joblib实现多进程并行的方案。那么对于用户来说，pickle序列化的性能就很关键。考虑到`traj-dist-rs`里面的动态规划算法返回的类型都是一个Rust定义的`PyDpResult`，那么对这个类型的序列化性能做提升就很重要。

先说一下这里是怎么设计的：

1. 考虑到动态规划算法的返回值有一个结果，还可能有完整的动态规划矩阵用于回溯路径，`traj-dist-rs`定义了一个`DpResult`的struct封装了这两个值。
2. 考虑到`traj-dist-rs`是同时支持Rust和Python两种语言的，并且提供了`python-binding`这个feature，用于额外编译适配Python的函数和类型。因此直接将`DpResult`暴露给Python环境不合理，因此需要再定义一个`PyDpResult`的struct暴露给python，而它只有一个叫inner的属性，类型是`DpResult`，这样Rust用户用的是`DpResult`，而python用户用的是`PyDpResult`，两者互不干扰，充分解耦。
3. 那么用户在python多进程环境中运行动态规划类算法的时候，就会面临返回值序列化和反序列化的问题，也就是`PyDpResult`的序列化和反序列化，这里很简单，实现`__reduce__`接口即可，那要把什么东西返回给python呢，其实就是把`DpResult`这个struct序列化为字节，传递给python，python拿到字节再反序列化为`DpResult`，然后创建一个`PyDpResult`将其封装即可。

因此这里最大的开销就是对`DpResult`的序列化，这里使用bincode，将其序列化为字节，这是我找到的最快的方案，相比serde_json肯定是快很多的。如果大家有其他方案也可以和我交流。

下面的源码是`PyDpResult`的代码，省略了一些与本文无关的内容，展示了如何进行序列化和反序列化。

```rust
/// Python wrapper for the Rust DpResult struct
///
/// This class wraps the Rust DpResult and provides Python-friendly access
/// to the distance and optional matrix.
#[cfg(feature = "python-binding")]
#[gen_stub_pyclass]
#[pyclass(name = "DpResult")]
pub struct PyDpResult {
    /// The inner Rust DpResult
    pub inner: crate::distance::DpResult,
}

#[cfg(feature = "python-binding")]
#[gen_stub_pymethods]
#[pymethods]
impl PyDpResult {
    /// Pickle serialization support using __reduce__
    ///
    /// Uses bincode to serialize the entire DpResult::inner as bytes for better performance.
    /// Returns a tuple (callable, args) that pickle can use to reconstruct the object.
    fn __reduce__(&self, py: Python) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        use pyo3::prelude::*;
        use pyo3::types::{PyBytes, PyTuple};

        // Import the module and get the helper function
        let module = py.import("traj_dist_rs")?;
        let helper_func = module.getattr("__dp_result_from_pickle")?;

        // Serialize the entire DpResult using bincode
        let serialized =
            bincode::encode_to_vec(&self.inner, bincode::config::standard()).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization failed: {}", e))
            })?;

        // Create args tuple containing the serialized bytes
        let bytes_py = PyBytes::new(py, &serialized);
        let args_tuple = PyTuple::new(py, [bytes_py.as_any()])?;

        // Return (helper_func, args, state) where state is None
        Ok((helper_func.unbind(), args_tuple.unbind().into(), py.None()))
    }
}

/// Helper function to create DpResult from pickle data
///
/// Deserializes the DpResult from bincode-encoded bytes.
#[cfg(feature = "python-binding")]
#[gen_stub_pyfunction]
#[pyfunction]
pub fn __dp_result_from_pickle(
    #[gen_stub(override_type(type_repr = "bytes"))] data: &[u8],
) -> PyResult<PyDpResult> {
    bincode::decode_from_slice(data, bincode::config::standard())
        .map(|(dp_result, _)| PyDpResult { inner: dp_result })
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Deserialization failed: {}", e))
        })
}
```

这样实现的序列化接口，就可以使`PyDpResult`达到一个比较好的序列化性能了，这样也就可以帮助Python多进程环境下的性能提升了。

## 六、实战：`traj-dist-rs`提升`TrajCL`的数据预处理性能

`TrajCL`是一个用于使用深度学习算法近似轨迹相似度的方法："TrajCL: Contrastive Trajectory Similarity Learning with Dual-Feature Attention"。在论文作者的[开源代码](https://github.com/changyanchuan/TrajCL/blob/master/utils/preprocessing_porto.py)里面，提供了对porto数据集中轨迹相似度的计算，使用`traj-dist`+python多进程完成。我们这里将作者的源代码取出并作轻量优化，与我们的`traj-dist-rs`的`pdist`并行版做性能对比。

作者从porto数据集中挑选了轨迹长度在20到200之间的序列，选出7000条作为训练集，计算这7000条矩阵之间的距离（上三角）。

我创建了一个4核8G的pod，对这7000条轨迹进行dtw计算，测试下来的效果如下：

|tool        |running time(s)|speedup  |
|------------|---------------|---------|
|traj-dist   |3111.451s      | 1x      |
|traj-dist-rs|165.745s       | 18.773x |

使用`traj-dist-rs`的`pdist`函数，可以让之前的数据预处理部分性能提升18.8倍，效果提升很明显。

下面是`traj-dist-rs`测试用的源码：

```python
import time
import polars as pl
from traj_dist_rs import Metric, pdist


def get_trajs():
    df = pl.read_parquet("trajcl_samples.parquet")
    start_idx, end_idx = 0, 7000

    trajs = [df["seq"][idx].to_numpy() for idx in range(start_idx, end_idx)]
    return trajs


if __name__ == "__main__":
    trajs = get_trajs()
    t = time.time()
    pdist(trajs, metric=Metric.dtw(), parallel=True).shape
    print(time.time() - t)
```

下面这是从论文作者代码中截取出来，并做了一些性能优化的代码：

```python
import time
import math
import multiprocessing as mp
from typing import List
import pandas as pd
import polars as pl
import traj_dist.distance as tdist


def _simi_matrix(fn, df) -> List[List[float]]:
    length = df.shape[0]
    batch_size = 50
    assert length % batch_size == 0

    tasks = []
    for i in range(math.ceil(length / batch_size)):
        if i < math.ceil(length / batch_size) - 1:
            tasks.append((fn, df, list(range(batch_size * i, batch_size * (i + 1)))))
        else:
            tasks.append((fn, df, list(range(batch_size * i, length))))

    num_cores = int(mp.cpu_count())

    pool = mp.Pool(num_cores)
    lst_simi = pool.starmap(_simi_comp_operator, tasks)
    pool.close()

    return lst_simi


def _simi_comp_operator(fn, df_trajs: pd.DataFrame, sub_idx: List[int]):
    simi = []
    length = df_trajs.shape[0]
    for _i in sub_idx:
        t_i = df_trajs.iloc[_i].seq
        simi_row = []
        for _j in range(_i + 1, length):
            t_j = df_trajs.iloc[_j].seq
            simi_row.append(fn(t_i, t_j))
        simi.append(simi_row)
    return simi


def get_train_df():
    df = pl.read_parquet("trajcl_samples.parquet")
    start_idx, end_idx = 0, 7000
    trajs = [df["seq"][idx].to_numpy() for idx in range(start_idx, end_idx)]
    df = pd.DataFrame(trajs, columns=["seq"])
    return df


if __name__ == "__main__":
    df = get_train_df()
    t = time.time()
    _simi_matrix(tdist.dtw, df)
    print(time.time() - t)
```

## 七、总结

这次批量计算的性能优化过程中，有几个宝贵的经验：

1. 瓶颈转移：优化完核心算法（Rust/Cython）后，瓶颈会转移到Python的调用层（循环、GIL、数据复制）。
2. “批处理”下沉：对于批量计算任务，最好的方式是设计一个能接收整个数据集的底层函数，将循环和调度完全下沉到高性能语言（Rust/C++）中。
3. 并行模型的选择：在Rust中，基于共享内存的多线程并行通常更优于Python中基于序列化的多进程并行，尤其是在数据量大时。
4. 并行化不是万金油：并行化有其自身开销。对于计算量极小的任务，串行就是最好的方案。在设计的时候就应该给用户提供串行和并行的接口，让用户自己选择。
