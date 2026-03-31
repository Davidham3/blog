---
title: "GEO 逼着我又给 Traj Dist Rs 做了一次性能提升"
description: 
date: 2026-03-31T14:29:23Z
image: 
math: true
license: 
hidden: false
comments: true
draft: false
---

最近一直在搞 `traj-dist-rs`，本来也没想拖这么久，早就应该发布 `1.0.0`，然后转去开发我另一个很重要的项目，但是总有些新发现，搞得我也有点头大。

今天我吃饱了撑的，非得去 Google 搜索一下 `trajectory similarity rust`，结果 Google 的 Gemini 说目前这个领域有两个强大的库，一个就是我开发的 `traj-dist-rs`，另一个是 `rapidgeo`。

我把内容粘贴过来了：

> In Rust, the primary crates for calculating trajectory similarity or distance measures are traj-dist-rs and rapidgeo-similarity.
>
> **Key Crates**
>
> - **traj-dist-rs**: This is a high-performance library providing a wide range of trajectory distance and similarity measures. It is designed for general-purpose trajectory analysis and includes Python bindings for interoperability.
>   - Algorithms included: Dynamic Time Warping (DTW), Longest Common Subsequence (LCSS), Edit Distance on Real Sequence (EDR), Edit Distance with Real Penalty (ERP), and others.
>   - Use cases: Trajectory clustering, similarity search (kNN), anomaly detection, and mobility data analysis.
>   - Similarity vs. Distance: The library primarily computes distances, which can be converted to similarity scores using functions like `Similarity = 1 / (1 + distance)` or Gaussian kernels.
>
> - **rapidgeo-similarity**: This crate focuses on fast similarity measures specifically for geographic polylines (GPS data).
>   - Algorithms included: Fréchet distance and Hausdorff distance, which are useful for geographic routes where point order is important.
>   - Features: Includes batch processing and parallel computation capabilities.

上面这里我干的事情，本质上就是利用最新的LLM与RAG系统，去搜索一个领域内的知识，而围绕着这个模式，现在有个叫**Generative Engine Optimization**的概念很火爆。

和传统 SEO 主要面向搜索引擎排序不同，GEO 关注的是：当用户直接向 Gemini、ChatGPT、Perplexity 这类生成式引擎提问时，一个项目会不会被提到、如何被描述，以及会和哪些项目一起被比较。

从这个角度看，开源项目现在不仅要面对 GitHub 上的 star、issue 和 benchmark，也要面对生成式引擎给出的“生态印象”。  

我这次就是在 Google 搜索 `trajectory similarity rust` 时，看到了 Gemini 把 `traj-dist-rs` 和 `rapidgeo` 并列，才临时起意去做了一轮 benchmark，结果一路顺着排查到了这次的性能优化。

因为我还是希望把我做得这个东西做成这个领域最强的库，所以我就突然脑子一抽，决定去和 `rapidgeo` 比一比，看看谁厉害。我就让 openclaw 给我写了个两个库性能对比的 Rust 代码，这里我就不粘贴了，一共 300 行，粘贴过来也麻烦。

但本质上就是 openclaw 去研究后发现，`rapidgeo` 只支持 `discrete frechet` 和 `hausdorff` 这两个算法。前者是一个动态规划算法，后者是一个比较暴力的笛卡尔积算法。然后 `rapidgeo` 只支持了 `discrete frechet` 的并行计算，`hausdorff` 没支持，所以从功能上来看比 `traj-dist-rs` 要差很多。

但是我也没管这些事情，就直接比一下就好了。

#### 单对轨迹 benchmark

| algorithm        |   length |   traj-dist-rs (s) |   rapidgeo (s) |   speedup |
|:-----------------|---------:|-------------------:|---------------:|----------:|
| discrete_frechet |       10 |           1.3e-05  |       1.3e-05  |  1.0722   |
| discrete_frechet |       50 |           0.000313 |       0.000322 |  1.02775  |
| discrete_frechet |      100 |           0.001034 |       0.001059 |  1.02443  |
| discrete_frechet |      500 |           0.030244 |       0.031647 |  1.04639  |
| hausdorff        |       10 |           6.5e-05  |       2e-05    |  0.305096 |
| hausdorff        |       50 |           0.001715 |       0.000448 |  0.261214 |
| hausdorff        |      100 |           0.007198 |       0.001891 |  0.262714 |
| hausdorff        |      500 |           0.189247 |       0.048976 |  0.258795 |

#### 批量 benchmark（并行版）

| algorithm        |   length |   traj-dist-rs par (s) |   rapidgeo (s) |   speedup |
|:-----------------|---------:|-----------------------:|---------------:|----------:|
| discrete_frechet |       10 |               0.006438 |       0.006031 |  0.936887 |
| discrete_frechet |       50 |               0.014179 |       0.016481 |  1.16233  |
| discrete_frechet |      100 |               0.066931 |       0.044106 |  0.658983 |
| discrete_frechet |      500 |               0.715868 |       0.521216 |  0.72809  |
| hausdorff        |       10 |               0.007766 |       0.003832 |  0.493439 |
| hausdorff        |       50 |               0.118687 |       0.092206 |  0.776887 |
| hausdorff        |      100 |               0.263854 |       0.373953 |  1.41727  |
| hausdorff        |      500 |               3.40011  |      10.1613   |  2.98851  |

在单对轨迹 `discrete frechet` 的测试中，`traj-dist-rs` 在所有测试长度下都略快于 `rapidgeo`。

但在单对轨迹的 `Hausdorff` 测试中，`traj-dist-rs` 明显落后于 `rapidgeo`，差不多 3 到 4 倍。

在批量并行 `discrete frechet` 的测试中，两者表现比较接近，但优势并不稳定。`traj-dist-rs` 只在长度 50 的设置下略占优势，而在更长轨迹上 `rapidgeo` 仍然更快。

当然，这只是很简单地比了一下，都没有上很复杂、很全面的对比。这已经可以看出来，至少在 `hausdorff` 这个算法的效率上，`traj-dist-rs` 有点慢了，那就只能去看一下问题到底出在哪里了。

我大概按三个方向去排查：

1. 先看并行实现的区别
2. 看 `discrete frechet` 和 `hausdorff` 的实现区别
3. 看底层 `haversine` 算法的实现区别

并行这里我去看了，没有太大区别，所以下面就不讲了，下面会针对最核心的算法实现讲解。

## 一、Haversine 的实现区别：空间换时间

这里我们先看 `haversine` 的实现区别。

深入研究后发现，`rapidgeo` 在计算 `haversine` 的时候，采用了空间换时间的策略。

我们从最朴素的 `lng1, lat1, lng2, lat2` 出发。

### 1.1 两点 Haversine 到底算了什么

先假设输入单位是“度”：

- 点1：`(lng1, lat1)`
- 点2：`(lng2, lat2)`

Haversine 常见写法：

$$
\phi_1 = lat1 \cdot \pi/180,\quad \phi_2 = lat2 \cdot \pi/180
$$

$$
\lambda_1 = lng1 \cdot \pi/180,\quad \lambda_2 = lng2 \cdot \pi/180
$$

$$
\Delta \phi = \phi_2 - \phi_1,\quad \Delta \lambda = \lambda_2 - \lambda_1
$$

$$
a=\sin^2(\Delta \phi /2)+\cos(\phi_1)\cos(\phi_2)\sin^2(\Delta \lambda /2)
$$

$$
d = 2R\arcsin(\sqrt{a})
$$

其中：

- `lat -> φ`
- `lng -> λ`
- `R` 是地球半径

假设现在只算**这一对点一次**，那其实看不出什么，因为每个量都只用一次。

但如果一个点会参与很多次比较，比如点1会和很多点比较，那就可以看出，我们需要对点1的经度和纬度反复做“度转弧度”，还有 `cos(纬度的弧度)`。做多少次呢？取决于点1到底要和多少个点比较，比较多少次，那就做多少次。

如果我们扩展到动态规划类算法，或者笛卡尔积算法，这类算法的时间复杂度一般是 `O(NM)`，也就是要做 `N` 个点与 `M` 个点之间的距离计算，那上面的“度转弧度”和 `cos(纬度的弧度)`，就要做 `N * M` 次。

那答案就显而易见了，我们做一下预计算就好了。也就是拿到两条轨迹后，先把 `lng_rad`、`lat_rad`、`cos(lat_rad)` 这三项算好。也就是给定两条经纬度序列（也就是 2 个矩阵），预处理的时候得到 6 个向量，后续所有球面距离的计算直接依赖这 6 个向量。这一部分的时间复杂度就从 `O(NM)` 下降到了 `O(N + M)`。

假设已经预处理好了：

- 点1：`(lng1_rad, lat1_rad, cos_lat1)`
- 点2：`(lng2_rad, lat2_rad, cos_lat2)`

那么距离计算只剩：

$$
\Delta \phi = lat2\_rad - lat1\_rad
$$

$$
\Delta \lambda = lng2\_rad - lng1\_rad
$$

$$
a=\sin^2(\Delta \phi /2)+cos\_lat1 \cdot cos\_lat2 \cdot \sin^2(\Delta \lambda /2)
$$

$$
d = 2R\arcsin(\sqrt{a})
$$

也就是说，每次点对距离还需要算：

- 2 次减法
- 2 次 `sin`
- 1 次乘法链
- 1 次 `sqrt`
- 1 次 `asin`

但不再需要：

- 4 次度转弧度
- 2 次 `cos(lat_rad)`

而 `rapidgeo` 的性能秘诀也就在这里。它为经度和纬度预计算了弧度，`cos(lat_rad)` 没有算，但也比我当时的实现要快很多了。

### 1.2 在 traj-dist-rs 里实现这个优化

那接下来就是在 `traj-dist-rs` 里面如何实现这个优化了。

实际上，架构设计永远是软件设计上的难题之一。

在当前的 `traj-dist-rs` 里面，支持 DP 算法和非 DP 算法，DP 算法抽象出了统一的框架。

#### 1.2.1 DP 算法的统一抽象

我们先说 DP 的部分。对于 `traj-dist-rs` 里面的大部分 DP 算法，我们只需要一个计算器，提供一个 `dis_between(idx1: usize, idx2: usize) -> f64` 的方法，获得轨迹 A 中点 `idx1` 与轨迹 B 中点 `idx2` 之间的距离，然后实现对应 DP 算法的状态转移方程即可。

因此 `traj-dist-rs` 定义了这个 trait：

```rust
pub trait DistanceCalculator {
    /// Calculate the distance between corresponding elements in two sequences
    fn dis_between(&self, seq_a_idx: usize, seq_b_idx: usize) -> f64;

    /// Calculate the distance between a point in a sequence and an external "anchor" point
    /// seq_id: 0 indicates the first sequence, 1 indicates the second
    fn compute_dis_for_extra_point<C: AsCoord>(
        &self,
        seq_id: usize,
        point_idx: usize,
        anchor: Option<&C>,
    ) -> f64;

    /// Get the length of the first sequence
    fn len_seq1(&self) -> usize;

    /// Get the length of the second sequence
    fn len_seq2(&self) -> usize;
}
```

有了这个 trait，常见的 DP 算法都可以被轻松实现。而且用户想用 `euclidean` 还是 `haversine`，只要封装在 `dis_between` 这个方法里面就可以了（`compute_dis_for_extra_point` 同理）。

`traj-dist-rs` 提供了对这个 trait 的一个具体实现：

```rust
/// Distance calculator based on trajectories
pub struct TrajectoryCalculator<'a, T, U>
where
    T: CoordSequence + 'a,
    U: CoordSequence + 'a,
{
    traj1: &'a T,
    traj2: &'a U,
    metric: DistanceType,
    /// Cache for spherical distance calculations (only populated when metric == Spherical)
    cache1: Option<SphericalTrajectoryCache>,
    cache2: Option<SphericalTrajectoryCache>,
}

impl<'a, T, U> TrajectoryCalculator<'a, T, U>
where
    T: CoordSequence + 'a,
    U: CoordSequence + 'a,
{
    /// Create a new trajectory calculator
    ///
    /// For spherical distance, this automatically precomputes intermediate values
    /// (lat/lon in radians and cos(lat)) to optimize subsequent distance calculations.
    pub fn new(traj1: &'a T, traj2: &'a U, metric: DistanceType) -> Self
    where
        T::Coord: AsCoord,
        U::Coord: AsCoord,
    {
        let (cache1, cache2) = match metric {
            DistanceType::Spherical => (
                Some(SphericalTrajectoryCache::from_trajectory(traj1)),
                Some(SphericalTrajectoryCache::from_trajectory(traj2)),
            ),
            DistanceType::Euclidean => (None, None),
        };

        Self {
            traj1,
            traj2,
            metric,
            cache1,
            cache2,
        }
    }
}

impl<'a, T, U> DistanceCalculator for TrajectoryCalculator<'a, T, U>
where
    T: CoordSequence + 'a,
    U: CoordSequence + 'a,
    T::Coord: AsCoord,
    U::Coord: AsCoord,
{
    fn dis_between(&self, idx1: usize, idx2: usize) -> f64 {
        match self.metric {
            DistanceType::Euclidean => {
                let p1 = self.traj1.get(idx1);
                let p2 = self.traj2.get(idx2);
                euclidean_distance(&p1, &p2)
            }
            DistanceType::Spherical => {
                // Use cached values for optimal performance
                let cache1 = self
                    .cache1
                    .as_ref()
                    .expect("Spherical cache not initialized");
                let cache2 = self
                    .cache2
                    .as_ref()
                    .expect("Spherical cache not initialized");
                great_circle_distance_cached(cache1, idx1, cache2, idx2)
            }
        }
    }

    fn compute_dis_for_extra_point<C: AsCoord>(
        &self,
        seq_id: usize,
        idx: usize,
        anchor: Option<&C>,
    ) -> f64 {
        let anchor = anchor.expect("anchor must not be None");
        match seq_id {
            0 => {
                let p = self.traj1.get(idx);
                self.metric.distance(anchor, &p)
            }
            1 => {
                let p = self.traj2.get(idx);
                self.metric.distance(anchor, &p)
            }
            _ => panic!("Invalid seq_id"),
        }
    }

    fn len_seq1(&self) -> usize {
        self.traj1.len()
    }

    fn len_seq2(&self) -> usize {
        self.traj2.len()
    }
}
```

可以看到，在这次升级中，增加了两个 `Option<SphericalTrajectoryCache>` 类型的 cache，这里面就包含了我们上文从算法层面讨论的 `lng_rad`、`lat_rad` 和 `cos(lat_rad)`。而 `dis_between` 方法只需要利用这两个类型的值，就可以轻松计算优化后的球面距离，以此实现性能提升。

#### 1.2.2 非 DP 算法的处理方式

那么对于非 DP 算法呢？

非 DP 算法因为有可能要计算轨迹 A 中某个点到轨迹 B 中某个线段的距离，所以没有强行使用上面的 `DistanceCalculator` 来统一接口。

因此非 DP 算法一般都是独立实现自身的计算过程。

也很简单：

```rust
pub struct SphericalTrajectoryCache {
    /// Longitude in radians
    lng_rad: Vec<f64>,
    /// Latitude in radians
    lat_rad: Vec<f64>,
    /// cos(latitude in radians)
    cos_lat: Vec<f64>,
}

impl SphericalTrajectoryCache {
    /// Create a new cache from a trajectory
    ///
    /// Precomputes all intermediate values needed for Haversine calculations.
    pub fn from_trajectory<T: CoordSequence>(traj: &T) -> Self
    where
        T::Coord: AsCoord,
    {
        let n = traj.len();
        let mut lng_rad = Vec::with_capacity(n);
        let mut lat_rad = Vec::with_capacity(n);
        let mut cos_lat = Vec::with_capacity(n);

        for i in 0..n {
            let p = traj.get(i);
            let lat_r = RAD * p.y();
            lng_rad.push(RAD * p.x());
            lat_rad.push(lat_r);
            cos_lat.push(lat_r.cos());
        }

        Self {
            lng_rad,
            lat_rad,
            cos_lat,
        }
    }

    /// Get the length of the cached trajectory
    pub fn len(&self) -> usize {
        self.lat_rad.len()
    }

    /// Check if the cached trajectory is empty
    pub fn is_empty(&self) -> bool {
        self.lat_rad.is_empty()
    }
}
```

这是上面提到的 `SphericalTrajectoryCache` 的源码。通过 `from_trajectory` 直接从泛型的轨迹数据就可以构建出来，再配合下面优化后的球面距离计算公式，就搞定了。

```rust
#[inline]
pub fn great_circle_distance_cached(
    cache1: &SphericalTrajectoryCache,
    i: usize,
    cache2: &SphericalTrajectoryCache,
    j: usize,
) -> f64 {
    let lat1 = cache1.lat_rad[i];
    let lon1 = cache1.lng_rad[i];
    let lat2 = cache2.lat_rad[j];
    let lon2 = cache2.lng_rad[j];

    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let sin_dlat_half = (dlat * 0.5).sin();
    let sin_dlon_half = (dlon * 0.5).sin();
    let a = sin_dlat_half * sin_dlat_half
        + cache1.cos_lat[i] * cache2.cos_lat[j] * sin_dlon_half * sin_dlon_half;
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    R * c
}
```

如果遇到再复杂一点的，也就是上面提到的轨迹 A 的点到轨迹 B 某个线段的距离计算，那就需要再下钻到每个细节的公式实现，比如点到线段的距离如何通过上面的缓存数据计算，这里就不展开介绍了。

### 1.3 优化后的 benchmark 结果

`traj-dist-rs` 在实现了上面的算法优化之后，确实在球面距离的计算上获得了比较明显的性能提升。

#### 单对轨迹 benchmark

| algorithm        |   length |   traj-dist-rs (s) |   rapidgeo (s) |   speedup |
|:-----------------|---------:|-------------------:|---------------:|----------:|
| discrete_frechet |       10 |           1e-05    |       1.2e-05  |  1.10626  |
| discrete_frechet |       50 |           0.000258 |       0.00029  |  1.12185  |
| discrete_frechet |      100 |           0.001013 |       0.001105 |  1.09074  |
| discrete_frechet |      500 |           0.0223   |       0.024643 |  1.10508  |
| hausdorff        |       10 |           6.2e-05  |       1.9e-05  |  0.312021 |
| hausdorff        |       50 |           0.001583 |       0.000443 |  0.279739 |
| hausdorff        |      100 |           0.005792 |       0.001451 |  0.250445 |
| hausdorff        |      500 |           0.161288 |       0.040726 |  0.252505 |

#### 批量 benchmark（并行版）

| algorithm        |   length |   traj-dist-rs par (s) |   rapidgeo (s) |   speedup |
|:-----------------|---------:|-----------------------:|---------------:|----------:|
| discrete_frechet |       10 |               0.00174  |       0.001759 |   1.01086 |
| discrete_frechet |       50 |               0.005438 |       0.006363 |   1.17003 |
| discrete_frechet |      100 |               0.018135 |       0.020553 |   1.13331 |
| discrete_frechet |      500 |               0.403519 |       0.473694 |   1.17391 |
| hausdorff        |       10 |               0.001961 |       0.00362  |   1.84601 |
| hausdorff        |       50 |               0.03089  |       0.083049 |   2.68849 |
| hausdorff        |      100 |               0.117196 |       0.343447 |   2.93054 |
| hausdorff        |      500 |               3.06087  |       7.87834  |   2.57389 |

可以看到，性能有了明显提升，`discrete frechet` 的性能基本上是追平了，并且更优；`hausdorff` 仍然有比较大的差距。

## 二、discrete Frechet 和 Hausdorff 的实现区别：算法定义不一样

`traj-dist-rs` 的性能更优，实际上就是我们对 DP 算法做了抽象得到的。`traj-dist-rs` 里面所有的 DP 算法默认都用了压缩空间后的 DP 算法，也就是只使用 2 行来存储中间状态；`rapidgeo` 用的是完整的 DP 矩阵，性能差距就是这里带来的。

而 `hausdorff` 出现了不同。仔细看了源码后发现，`rapidgeo` 实现的是离散 `hausdorff`，也就是计算两条轨迹里面任意两个点之间的距离，然后用 Hausdorff 的算法，采用先取 `min`，再取 `max` 的思路。

而 `traj-dist-rs` 由于是复刻 `traj-dist`，采用的是轨迹 A 里面的点到轨迹 B 里面所有连续线段之间的距离，然后再取 `min` 和 `max`。

这就意味着，`traj-dist-rs` 的 `hausdorff` 计算量一定会比 `rapidgeo` 的更大，因为点到线段的距离计算肯定会涉及到投影等复杂运算。

这也注定了即便是优化了底层的 `haversine` 计算，`traj-dist-rs` 在 `hausdorff` 上的性能也很难超过 `rapidgeo`。在上面的表格里之所以看到并行计算时 `traj-dist-rs` 更快，那是因为 `rapidgeo` 没有给 `hausdorff` 设计并行计算的接口，这里是通过 `for` 循环实现的，所以完全不具备参考意义。

不过，`traj-dist-rs` 是可以支持像 `rapidgeo` 这样的离散 `hausdorff` 的。不过考虑到现在是 `1.0.0-rc.3` 的开发阶段，可以把这个内容留到 `1.1.0` 再做。

## 三、总结

一次简单的心血来潮，还是让我看到了新的东西。

我之前确实思考过 `haversine` 能不能优化，因为 `traj-dist-rs` 开发出来后发现 Euclidean 上的性能比 `traj-dist` 强很多，但是 `haversine` 就没有强很多。当时我认为，三角函数的计算必然是计算密集型任务，性能提升倍数没有那么多很正常，毕竟 `traj-dist` 的内核是 Cython，然后我就没再思考过这个问题了。

结果今天的一个简单对比，直接学到了针对这种 pair 对计算的场景，可以做预计算的优化思路。虽然这个思路并不深奥，而且我之前在其他场景也用到过，但这次确实是没想到。

然后就是发现了 `hausdorff` 的实现上确实有区别。实际上还是应该去看论文，看一下使用 `hausdorff` 衡量轨迹相似度的论文，到底用的是哪种实现。

当然，这次优化还是带来了显著的性能提升的，因为traj-dist-rs不像rapidgeo只支持2个算法，而是支持7个算法，因此1次在架构上的增强，直接让7个算法、连带整个并行模块都拿到了收益。

整体来看，traj-dist-rs在haversine的距离上比traj-dist的cython实现的性能提升从3.1倍提升到了3.6倍。

- DTW (spherical): 2.97x → 3.40x (+14%)
- SSPD (spherical): 1.72x → 2.25x (+31%)
- Hausdorff (spherical): 2.41x → 2.91x (+21%)
- LCSS (spherical): 2.68x → 3.00x (+12%)
- EDR (spherical): 2.58x → 3.32x (+29%)
- ERP (spherical): 6.30x → 6.77x (+7%)

每个独立的算法也有明显提升，这里没写discrete frechet是因为traj-dist不支持haversine距离的discrete frechet算法。

- Batch `cdist` spherical (length=1000): 7.73x → **12.01x** (+55%)
- Batch `pdist` spherical (length=1000): 10.38x → **17.81x** (+72%)

两个并行接口也有明显的性能提升，收益很大。

## Reference

1. [traj-dist-rs](https://github.com/Davidham3/traj-dist-rs)
2. [rapidgeo](https://github.com/gaker/rapidgeo)
