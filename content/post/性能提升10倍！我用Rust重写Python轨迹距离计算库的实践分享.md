---
date: '2026-02-08T23:18:49Z'
draft: false
tags:
- trajectory
- rust
title: '性能提升10倍！我用Rust重写Python轨迹距离计算库的实践分享'
---

最近在做轨迹数据分析相关的项目，需要计算轨迹之间的距离。一开始用的是 `traj-dist` 这个 Python 包，它提供了多种轨迹距离算法的实现，速度很快，因为是通过Cython加速的。但是这个包有点太老了，安装起来很麻烦。

于是我想着能不能用 Rust 重写一遍，一是为了适配更新的python版本，二是为了尝试“用rust重写一切”。

实现完了之后测试了一下性能，吓了一跳：相比 `traj-dist` 的 Python 实现，性能提升了 389 倍，相比 Cython 实现也提升了 10 倍。本文我会分享一下这个项目的一些经验和踩过的坑。

<!--more-->

## 一、为什么要重写

`traj-dist` 是一个挺不错的包，提供了多种主流的轨迹距离算法：

- SSPD（对称段路径距离）
- DTW（动态时间规整）
- Hausdorff 距离
- LCSS（最长公共子序列）
- EDR（实数序列编辑距离）
- ERP（带实数惩罚的编辑距离）
- Discret Fréchet 距离
- 还有很多，我还没有重写

支持欧几里得距离和球面距离（Haversine），API 设计得很简单。

但是，核心痛点是这个包的安装比较麻烦，我尝试过在python3.12上安装，但是发现它的依赖太老。而且每次安装都需要编译Cython代码。为此我希望有一个安装简单，而且依赖比较少的包出现。

## 二、重写思路

我先说一下我的重写思路：
1. 确定需要重写的算法：首先我在`traj-dist`里面挑选了一些简单的算法：
    - LCSS、DTW、discrete frechet、EDR、ERP，这些都是动态规划算法
    - hausdorff，sspd，这两个算法是非动态规划的算法，实现起来相对简单
    - 其他算法的实现稍微复杂了一些，就留在后面的版本了
2. 以`traj-dist`为ground truth：我发现作者在代码仓库里面提供了一个很小的轨迹数据集，并且作者自己在上面进行了性能测试，那我们可以利用这个数据集，使用`traj-dist`在上面完成上面7个算法的计算，记录结果与耗时。
3. 使用rust重写上述7个算法，rust里面使用trait表示经纬度点和轨迹，为其他人使用这个包提供扩展性。同时为rust编写单元测试用例，保证其API都是可用的。
4. 为rust的算法和trait定义python相关的类型和算法，暴露给python。同时为python部分编写测试用例，一方面保证API可用，另一方面与上面的ground truth做对比，确保每个算法的计算结果与ground truth的结果偏差值在1e-8以内。
5. 由于ground truth里面带了`traj-dist`的性能，因此`traj-dist-rs`也可以很轻松地做出性能对比，得到rust相对于python和cython的性能提升比例。

## 三、性能测试结果

性能测试这里，首先随机选1000个轨迹pair，每个样本预热10次，正式测量50次，记录这50次的明细时间，并统计中位数、Coefficient of Variation等值。

因此会用`traj-dist`的python版本、cython版本，`traj-dist-rs`的python版本运行相同的测试用例，记录均值、标准差、CV值。

下面是具体的提升效果：

**EUCLIDEAN 距离**

- **Rust vs Cython**: 平均提升 9.99x (范围: 6.06x - 16.13x)
- Rust vs Python: 平均提升 389.07x (范围: 187.20x - 594.74x)
- **最佳性能提升算法**: sspd (16.13x)

**SPHERICAL 距离**

- **Rust vs Cython**: 平均提升 2.66x (范围: 1.62x - 5.24x)
- Rust vs Python: 平均提升 75.67x (范围: 41.13x - 167.14x)
- **最佳性能提升算法**: erp (5.24x)

这个结果比我预期的要好很多。相比 Python 实现提升 389 倍，这主要是因为 Rust 没有解释器开销，像动态规划算法这种需要双重for循环的算法，python一定是非常慢的，而且rust的编译器能做很多优化。相比 Cython 实现提升 10 倍，这部分主要是数据拷贝带来的，rust的实现中使用了numpy+零拷贝的方案，避免了数据复制，论计算性能来说，rust和cython生成的c代码，本质上应该不会有太大区别。

至于欧氏距离上的性能提升远大于球面距离的提升，这是因为欧氏距离的计算太简单，"语言效率"成为瓶颈，Rust 优势明显；球面距离的计算太复杂，"数学计算"成为瓶颈，Rust 优势被稀释。

## 四、架构设计

接下来说一下架构的设计，核心有几个内容吧：
1. rust接口设计
2. 动态规划算法优化
3. rust与python的双接口设计

### 4.1 rust接口设计

rust设计了几个关键的Trait：

```rust
pub trait AsCoord {
    /// Get the x-coordinate (longitude or easting)
    ///
    /// Returns the x-coordinate value of the point.
    fn x(&self) -> f64;

    /// Get the y-coordinate (latitude or northing)
    ///
    /// Returns the y-coordinate value of the point.
    fn y(&self) -> f64;
}
```

这个AsCoord用来表示一个经纬度点，因为肯定要计算经纬度点之间的距离，那么用这样一个trait抽象，不论用户是用`&[f64]`还是`vec`，都可以轻松支持，并且保证很好的性能。

```rust
pub trait CoordSequence {
    /// The type of coordinate in this sequence
    type Coord: AsCoord;

    /// Get the number of coordinates in the sequence
    ///
    /// Returns the total number of coordinate points in the sequence.
    fn len(&self) -> usize;

    /// Check if the sequence is empty
    ///
    /// Returns `true` if the sequence contains no coordinates, `false` otherwise.
    /// This is implemented as `self.len() == 0` by default.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the i-th coordinate in the sequence
    ///
    /// Returns a copy of the coordinate at the specified index.
    ///
    /// # Panics
    ///
    /// This method may panic if the index is out of bounds.
    fn get(&self, i: usize) -> Self::Coord;
}
```

这个trait表示轨迹，也就是经纬度序列。因此只要用户将自己的数据类型实现这个trait的接口，就可以调用`traj-dist-rs`里面的距离计算函数了。

### 4.2 动态规划算法优化

#### 4.2.1 动态规划算法的类型抽象

对于动态规划算法来说，一般只需要计算任意两个点之间的距离。考虑到用户需要使用的距离计算方法`traj-dist-rs`不支持，比如导航距离。因此需要给用户提供一个类似scikit-learn的方案：用户可用预先计算好任意两点之间的距离，然后传入这个“预计算距离矩阵”。

因此对于动态规划类的算法，`traj-dist-rs`提供两种接口形式：
1. 用户可用传入2条原始的经纬度序列。
2. 用户提供一个预计算距离矩阵。

针对这两种类型，我定义了一个叫`DistanceCalculator`的trait。

```rust
pub trait DistanceCalculator {
    /// Calculate the distance between corresponding elements in two sequences
    fn dis_between(&self, seq_a_idx: usize, seq_b_idx: usize) -> f64;

    /// Calculate the distance between a point in a sequence and an external "anchor" point
    fn compute_dis_for_extra_point<C: AsCoord>(
        &self,
        seq_id: usize,
        point_idx: usize,
        anchor: Option<&C>,
    ) -> f64;

    /// Length of sequence 1
    fn len_seq1(&self) -> usize;

    /// Length of sequence 2
    fn len_seq2(&self) -> usize;
}
```

动态规划算法完全可以通过这个trait的泛型，取出任意两点之间的距离，完成动态规划算法的计算。

当然，`traj-dist-rs`里面提供了两个具体的类型实现了这个trait。

```rust
pub struct TrajectoryCalculator<'a, T, U>
where
    T: CoordSequence + 'a,
    U: CoordSequence + 'a,
{
    traj1: &'a T,
    traj2: &'a U,
    metric: DistanceType,
}

pub struct PrecomputedDistanceCalculator<'a> {
    distance_matrix: &'a Vec<Vec<f64>>,
    seq1_extra_dists: Option<&'a Vec<f64>>,
    seq2_extra_dists: Option<&'a Vec<f64>>,
}
```

1. `TrajectoryCalculator`：直接访问轨迹数据，按需计算距离
2. `PrecomputedDistanceCalculator`：使用预计算的距离矩阵

这样的好处是算法逻辑和距离计算逻辑解耦，如果将来要支持其他距离计算方式，只需要实现新的 Calculator 就可以了。比如用户需要实时调用接口获取两点之间的导航距离，那么用户自己实现这个trait就可以使用动态规划算法了。

下面的代码是使用`traj-dist-rs`提供的`TrajectoryCalculator`调用动态规划算法的示例：

```rust
use traj_dist_rs::distance::{self, base::TrajectoryCalculator, distance_type::DistanceType};

let traj1 = vec![[0.0, 0.0], [1.0, 1.0]];
let traj2 = vec![[0.0, 1.0], [1.0, 0.0]];

let calculator = TrajectoryCalculator::new(&traj1, &traj2, DistanceType::Euclidean);
let dist = distance::dtw::dtw(&calculator, false);
```

#### 4.2.2 动态规划算法内存优化

`traj-dist-rs`针对动态规划算法还做了一个内存上的优化：对于大部分动态规划算法，当我们不需要回溯动态规划算法的路径的时候，其空间复杂度可以缩减到`O(MN) -> O(min{M, N})`，这里的M和N是2条序列的长度。`traj-dist-rs`会默认使用这种算法记录动态规划算法的中间值，实现内存使用的下降。

### 4.3 双接口支持

这个项目同时提供了 Rust API 和 Python 绑定。因为据我了解计算轨迹距离，大家基本都是python实现，而且学术界用得更多，因为轨迹表示学习之后的度量方式，很多论文会使用更传统的metrics，比如DTW，frechet等。

因此如果要用python做的话，cython这个加速方案有点老了，而且需要学习cython代码；pybind11与c++的方案也可以，但是不如pyo3+rust的方案。

如果是这样，我还不希望这个包只服务python用户，应该也能服务rust用户。这样就要求rust的代码应该更简单、抽象、内聚，在此基础上添加python的binding。

Python 绑定通过 PyO3 实现，API 设计尽量保持和 traj-dist 一致：

```python
import traj_dist_rs
import numpy as np

traj1 = np.array([[0.0, 0.0], [1.0, 1.0]])
traj2 = np.array([[0.0, 1.0], [1.0, 0.0]])

# 使用方式几乎和 traj-dist 一样
dist = traj_dist_rs.dtw(traj1, traj2, "euclidean")
dist = traj_dist_rs.lcss(traj1, traj2, "euclidean", eps=0.5)

# ERP 有两个版本
dist_standard = traj_dist_rs.erp_standard(traj1, traj2, "euclidean", g=[0.0, 0.0])
dist_compat = traj_dist_rs.erp_compat_traj_dist(traj1, traj2, "euclidean", g=[0.0, 0.0])
```

考虑到大部分数据都是在python中读取的，那么如何让rust通过零拷贝的方式读取是提升性能的一个关键。因为传统方式里面，当 Python 调用 C/C++/Cython 模块时，往往需要先把这箱货物“卸下”（从Python内存），再“搬运”并“重新打包”（复制到C/C++/Cython能理解的内存结构中）。这个过程是有开销的。

我这里利用了numpy的c-contiguous特性，当numpy.ndarray的数据在内存中是连续存储的时候，可以不需要“搬运”和“打包”操作，直接在原地打开货物。也就是说：rust可以直接读取python中的numpy.ndarray的数据，而不是复制一份。这里rust拿到的是一个`&[f64]`的切片。接下来只要把这个切片实现上面介绍的`CoordSequence`接口，就可以适配各个算法了。rust的零成本抽象可以直接让各个算法操作这块内存中的数据，以此实现性能提升，同时也避免了数据的拷贝，这也是rust性能超越cython的核心原因。

## 五、踩过的坑

做这个项目的过程中，也发现了一些问题。

### 5.1 球面距离的精度问题

一开始测试的时候，我发现球面距离计算的结果和 `traj-dist` 的结果对不上。排查了很久，最后发现问题出在 `traj-dist` 的 Cython 实现上。

traj-dist 的 Cython 代码中，球面距离计算使用的是 `float`（32 位浮点数），而且 PI 值是硬编码的截断值：

```cython
cdef double pi = 3.14159265
```

这个精度显然不够。为了和 `traj-dist` 的结果对比，我fork并修改了`traj-dist`的源码，把 `float` 改成 `double`，使用 `M_PI` 常量：

```cython
cdef double pi = M_PI
```

修复后，Rust 实现和 traj-dist 的结果就对上了。

### 5.2 ERP 算法的实现问题

在测试过程中，我发现 traj-dist 的 ERP 算法实现有问题。

ERP算法是典型的动态规划算法，在初始化动态规划矩阵的边界值的时候，也就是`C[0,:]`和`C[:,0]`，`traj-dist`使用了错误的方法，给每个值都初始化成了相同的值，这里的正确实现应该是不同的值。

这里我纠结了半天，考虑到这个包已经被很多人用了，直接修改 API 会破坏兼容性。

于是我想了个折中的方案：提供两个版本

- `erp_standard`：按照正确的算法实现
- `erp_compat_traj_dist`：按照 `traj-dist` 的问题版本实现，保持兼容性。

这样，新用户可以用正确的实现，老用户如果需要和原来保持一致，也可以用兼容版本。


## 六、精度保证

`traj-dist-rs`的实现，以`traj-dist`为基准，因此需要两者跑出来的结果尽可能一致。

这里我做了两件事：
1. 首先修改`traj-dist`的球面距离计算代码，将所有的float部分替换为double，并且修改PI的值为M_PI，以获得更高的精度。
2. 因为`traj-dist`的erp算法实现有误，为了保证结果的一致性，我这里测量的是`traj-dist-rs`的`erp_compat_traj_dist`与`traj-dist`的erp算法之间的误差。

测算下来误差远远小于我期望的1e-8的阈值，说明`traj-dist-rs`实现了算法的正确实现。

## 七、总结

`traj-dist-rs`的实现我觉得有几个亮点：
1. 使用rust重写
2. 与python的绑定中考虑了零拷贝的问题
3. rust层面做了比较好的抽象，不论是`CoordSequence`还是`DistanceCalculator`。
4. 所有动态规划算法提供了节省内存的实现。
5. 确实实现了性能的显著提升。

如果你也在做轨迹数据分析，或者对 Rust 性能优化感兴趣，欢迎试试 `traj-dist-rs`。项目地址在：[traj-dist-rs](https://github.com/Davidham3/traj-dist-rs)。目前alpha版本已经发布在crates.io和pypi，同时也支持context7。

有任何问题或建议，欢迎交流！

## 附录

1. 修复球面距离精度的：[traj-dist](https://github.com/Davidham3/traj-dist)。
2. 完整的性能测试记录：[performance.md](https://github.com/Davidham3/traj-dist-rs/blob/v0.1.0-alpha.3/docs/performance.md)。
