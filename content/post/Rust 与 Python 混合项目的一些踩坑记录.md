---
title: "Rust 与 Python 混合项目的一些踩坑记录"
description: 
date: 2026-03-23T14:38:41Z
image: 
math: 
license: 
hidden: false
comments: true
draft: false
---

最近这段时间，我一直在做一个叫 `traj-dist-rs` 的库。这是一个用于计算轨迹距离和轨迹相似度的库，核心计算部分由 Rust 实现，再通过 `pyo3` 将关键类型和函数暴露给 Python，这样 Rust 和 Python 用户都可以直接使用。

这个项目最初来自一个很实际的需求。

之前已经有一个叫 `traj-dist` 的 Python 库可以解决类似问题，但这个库诞生得比较早，而且多年没有维护，放到今天的 Python 生态里已经不太好用了。与其继续围绕旧实现打补丁，我最后选择直接重写一版，尽量贴近现在的工具链和使用习惯。

写这篇文章的时候，`traj-dist-rs` 已经发布到 `1.0.0-rc.2`。如果后续测试一切正常，正式的 `1.0.0` 应该也快了。

因为核心开发已经接近尾声，我想把这段时间学到的一些东西整理一下。内容主要是 Rust 与 Python 混合项目中的一些工程实践。当然，由于整个项目本身也是通过 AI Coding 辅助开发完成的，所以最后我也会补充一点我对 AI Coding 的理解和判断。

这篇文章大概分成四部分：

- 代码实现
- 文档建设
- 打包与发布
- 关于 AI Coding 的一些感想

---

## 一、代码部分

`traj-dist-rs` 的主体代码都写在 Rust 里，Python 层比较薄，主要负责把 Rust 的能力封装成一个可用的 Python 包。

这个思路本身并不复杂：性能敏感的部分交给 Rust，用户入口留给 Python。  
但真正实现的时候，会遇到不少只有在混合语言项目里才会明显暴露出来的问题。

---

### 1. 用 Rust feature 拆分核心库和 Python 绑定

这个项目同时面向两类用户：

- 直接在 Rust 里使用它的人
- 从 Python 调用它的人

这意味着，Python 绑定相关的依赖并不应该强加给所有构建。例如 `pyo3`、`numpy`、`pyo3-stub-gen` 这些 crate，如果只是把它当作 Rust 库来使用，其实完全没有必要引入。

Rust 的 `feature` 在这里就很好用。

我最后大概是这样组织的：

```toml
[features]
default = []
python-binding = ["pyo3", "pyo3-stub-gen", "pyo3-stub-gen-derive", "numpy"]
parallel = ["rayon", "num_cpus"]

[dependencies]
thiserror = "2"
strum = "0.26"
strum_macros = "0.26"
bincode = "2"
pyo3 = { version = "0.26", optional = true }
pyo3-stub-gen = { version = "0.17", optional = true }
pyo3-stub-gen-derive = { version = "0.17", optional = true }
numpy = { version = "0.26", optional = true }
rayon = { version = "1.8", optional = true }
num_cpus = { version = "1.16", optional = true }
```

对应地，Python 绑定部分的代码只在启用相关 feature 时才编译：

```rust
#[cfg(feature = "python-binding")]
```

这个做法带来的好处很直接：

- 核心算法代码和 Python 绑定代码边界很清晰
- 纯 Rust 用户不会额外编译一堆 Python 相关依赖
- 项目结构更容易长期维护

对于 Rust-Python 混合项目来说，这种拆分方式几乎没有什么额外负担，但收益非常稳定，而且实现成本也不高。

---

### 2. 零拷贝读取 NumPy 数组

做轨迹距离计算时，有一个很现实的问题：**坐标数据会被频繁读取**。

这类数据通常会以二维 `numpy.ndarray` 的形式存储，例如经纬度序列一般就是 `(N, 2)` 的数组。如果每次从 Python 传入一个 NumPy 数组，Rust 这边都先复制一份再进入计算，那么不仅性能会受影响，也会带来额外的内存开销。理想情况当然是：**Rust 直接读取 Python 传进来的底层数组内存。**

在做这个项目之前，我对 NumPy 的理解基本停留在“会用”的层面，知道 `shape`、`dtype`、广播这些概念，但并没有太在意底层存储方式。这个项目第一次让我认真去理解 `C-contiguous`。

如果一个二维 NumPy 数组是 C 连续的，那么它的元素在内存中就是按行连续排列的。这样 Rust 就可以安全地拿到指针，并把它当成一段连续的切片来读取。

假设有这样一个数组：

```python
import numpy as np

arr = np.array([
    [1.2, 5.3],
    [7.9, 2.1],
    [3.4, 8.8],
], dtype=np.float64)
```

从逻辑上看，它的 shape 是 `(3, 2)`，也就是 3 个点：

```text
[
  [1.2, 5.3],
  [7.9, 2.1],
  [3.4, 8.8]
]
```

如果它是 C-contiguous，那么底层内存实际上可以理解为这样一段连续区域：

```text
| 1.2 | 5.3 | 7.9 | 2.1 | 3.4 | 8.8 |
```

也就是说，Python 层看到的是二维结构，而底层内存本质上仍然是一段按行排列的一维连续数据。对于 Rust 来说，只要能够获取这段内存的起始指针，就可以按照约定的布局直接读取里面的元素。

我最后写了一个类似这样的包装：

```rust
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryRef<'a> {
    data_ptr: *const f64,
    len: usize,
    _phantom: PhantomData<&'a f64>,
}

impl<'a> TrajectoryRef<'a> {
    pub fn new(array: PyReadonlyArray2<'a, f64>) -> Result<Self, TrajDistError> {
        let shape = array.shape();
        if shape.len() != 2 || shape[1] != 2 {
            return Err(TrajDistError::DataConvertionError(format!(
                "Numpy array must have a shape of (N, 2), but got {:?}",
                shape
            )));
        }

        let view = array.as_array();
        if view.is_standard_layout() {
            Ok(Self {
                data_ptr: view.as_ptr(),
                len: shape[0],
                _phantom: PhantomData,
            })
        } else {
            Err(TrajDistError::DataConvertionError(
                "Numpy array must be contiguous (C-order)".to_string(),
            ))
        }
    }

    #[inline(always)]
    fn get_data(&self) -> &'a [f64] {
        unsafe { std::slice::from_raw_parts(self.data_ptr, self.len * 2) }
    }
}
```

然后再把单个点表示成对底层数据的一层轻量引用：

```rust
#[derive(Debug, Clone, Copy)]
pub struct PointRef<'a> {
    data: &'a [f64],
    idx: usize,
}

impl<'a> PointRef<'a> {
    pub fn new(data: &'a [f64], idx: usize) -> Self {
        Self { data, idx }
    }
}

impl<'a> AsCoord for PointRef<'a> {
    #[inline(always)]
    fn x(&self) -> f64 {
        self.data[self.idx * 2]
    }

    #[inline(always)]
    fn y(&self) -> f64 {
        self.data[self.idx * 2 + 1]
    }
}

impl<'a> CoordSequence for TrajectoryRef<'a> {
    type Coord = PointRef<'a>;

    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> Self::Coord {
        PointRef::new(self.get_data(), idx)
    }
}
```

这一块其实很能体现 Rust 的优势：我们可以在比较底层地处理指针、切片和生命周期的同时，把暴露给上层算法的接口做得非常干净。

这里本质上是借助 `numpy` crate 拿到底层指针，再通过实现自定义的 `AsCoord` 和 `CoordSequence` trait，把一段 NumPy 内存映射成轨迹坐标序列。这样上层算法仍然可以按照统一的抽象来写，而底层又保留了非常低的额外开销。

现在回头看，这部分应该是整个 `traj-dist-rs` 里我自己最满意的一块：实现并不复杂，但接口和性能都兼顾得比较好。

---

### 3. 不同环境下的 Rayon 线程池

我在项目里用了 `Rayon` 来做并行计算。在本地开发阶段，默认线程池的体验很好：自动初始化，线程数通常也会按 CPU 核数来决定，对单机程序来说非常省心。

问题出在分布式环境里，尤其是 Spark / PySpark 这种场景。

举个例子：一台机器可能有 64 个核，但上面会启动多个 executor，而每个 executor 实际只分到 4 个核。如果 Rayon 看到的是整台机器的 CPU 数，而不是当前 executor 真正拿到的资源，那它就可能初始化出一个远超预期的线程池。

结果就是典型的线程过度竞争：

- executor 之间互相抢 CPU
- 线程数看起来很多，但整体吞吐反而下降
- 实际表现甚至可能不如少开一点线程

这种问题在本地环境里几乎看不出来，但到了集群里就会变得很明显。对 CPU 密集型任务来说，默认配置未必就是合理配置。

最后最直接的解决办法反而很朴素：**显式控制 Rayon 的线程数**。

```python
import os
os.environ["RAYON_NUM_THREADS"] = "4"
```

这样至少可以保证 executor 用多少核，Rayon 就开多少线程，不会出现“每个进程都以为自己独占整台机器”的情况。

我之所以意识到这个问题，是因为有一天准备把这个库放到 Spark 环境里跑，突然想到：Rayon 的线程池到底会怎么工作？进一步想下去就会发现，Spark executor 的资源分配和 Rayon 默认线程池模型并不是天然对齐的，因此如果不额外控制线程数，就很容易出现过度竞争。

---

## 二、文档

文档这部分，我这次也换了一套思路。

以前我写文档，基本都是 **Sphinx + MyST**。这套组合功能很强，也很成熟，而且我之前有个叫 `TrajDL` 的项目，里面有大量 Jupyter Notebook 写的教程，都是通过 `MyST-NB` 渲染的，效果很好。

不过这次做 `traj-dist-rs`，我想试试别的方案，所以最后用了 **MkDocs + Mike**。

整体体验下来，我最直接的感受是：**轻很多，也省心很多。**

不是说它能力一定更强，而是对于这个项目来说，它刚好够用，而且没有让我把太多精力花在文档系统本身上。

---

### 1. MkDocs + Mike 比我预期中更适合这种项目

这次的文档需求其实比较明确：

- 要能放 API 文档
- 要能写教程和示例
- 要支持版本化发布
- 最好不要为了搭文档写太多额外配置

`MkDocs` 在这些点上都还不错，而 `Mike` 则刚好补上了版本化文档这一块。

最终的感觉就是：如果项目文档不是那种结构极其复杂的大型站点，那么 MkDocs 这类方案会非常轻量，很适合快速把内容组织起来。

另外，对于 Rust-Python 混合项目来说，`.pyi` stub 文件也能比较自然地融入文档链路，这一点对我来说很重要。编译型的扩展模块本身对 IDE 和文档工具都不够友好，而 stub 在这里不只是类型提示的问题，某种程度上也是文档的一部分。对我这个项目来说，API 文档的生成链路实际上也依赖这个文件。

---

### 2. notebook 写文档，比我预想中更实用

我以前喜欢 MyST 的一个重要原因，就是它和 notebook 的结合做得很好。很多技术文档，尤其是偏示例驱动的内容，用 notebook 写其实非常自然：

- 代码可以直接运行
- 输出结果就在旁边
- 调试、修改和迭代都比较顺手

这次换到 MkDocs 后，我本来担心这部分体验会变差，后来发现 `mkdocs-jupyter` 其实也能满足需求。更让我意外的是，Rust 的代码示例也可以放进 notebook 里直接执行，靠的是 `evcxr`。

安装方式很简单：

```bash
cargo install evcxr_jupyter
evcxr_jupyter --install
```

装好之后，Jupyter 就可以直接运行 Rust 代码了。

这个体验比我一开始预想的要自然很多。因为 `traj-dist-rs` 本来就是一个双语言项目，如果文档里的 Python 示例和 Rust 示例是两套完全割裂的体系，维护起来其实会比较痛苦。现在两边都能走 notebook 风格，整体上就统一多了。

最后，我在文档里的 Usage 部分大概变成了这样的工作流：

- 用 notebook 写和调试示例
- Python 示例直接运行
- Rust 示例也直接运行
- 最后通过 MkDocs 渲染成站点页面

我自己一直比较喜欢用 notebook 来写 usage 或 tutorial，可能也和我当初受《动手学深度学习》这个项目影响比较大有关。

---

### 3. 文档真的上线之后，才会开始关心“能不能被搜到”

因为我用的是 `Mike`，而它可以比较方便地把文档部署到仓库的 GitHub Pages 页面上，所以我自然也希望 Google 能搜到这些文档。

这时候就会开始涉及一些 SEO 相关的工作。整体并不复杂，大概就是下面几步：

1. 在 Google Search Console 里录入文档站点，例如 `latest` 页面
2. 按照 Google 提示，把验证 token 植入页面
3. 在 MkDocs 配置里启用自定义模板目录，例如在 `theme` 下设置 `custom_dir: docs/overrides`
4. 在这个目录里创建 `main.html`，插入 Google 提供的验证 meta 标签

类似这样：

```html
{% extends "base.html" %}

{% block extrahead %}
  <meta name="google-site-verification" content="{{google_token}}" />
{% endblock %}
```

配置好之后，Google 就可以开始收录这个网站了。通常过一段时间之后就能在搜索里找到。Bing 这边也相对简单，因为它支持导入 Google Search Console 的相关内容，所以顺手配置一下就够了。

这种事情在文档还没发布之前其实很容易被忽略，但一旦你真的把文档站上线，就会开始意识到：**“有文档” 和 “别人能找到文档” 是两回事。**

---

## 三、打包

如果说整个项目里哪一部分最容易低估时间成本，那大概率就是打包和发布。

算法写出来、绑定做好、测试跑通，这些都不代表“用户就能顺利用上”。  
真正要把一个 Rust-Python 混合库稳定地发成 wheel，细节会非常多。

---

### 1. 我的 `.pyi` 没被打包进 wheel

Rust-Python 混合库有一个很典型的问题：核心实现都在编译后的 `.so` 里，对 Python 来说当然可以 import、可以运行，但对 IDE 来说就不够友好了。

如果希望 VS Code、PyCharm 这类工具提供更好的自动补全、类型提示和函数签名，那么 stub 文件几乎是必需品。

我这里用 `pyo3-stub-gen` 生成了 `_lib.pyi`，因此 Python 源码目录结构大概是这样：

```text
python/
└── traj_dist_rs
    ├── __init__.py
    ├── _lib.cpython-310-x86_64-linux-gnu.so
    ├── _lib.pyi
    └── py.typed
```

按理说这件事应该很简单：生成 stub，打进 wheel，结束。

但实际情况并没有这么简单。我后来检查 wheel 内容时发现，`_lib.pyi` 根本没有被打进去。

这个问题最烦的地方在于，它看起来并不像“构建失败”。wheel 能正常产出，扩展模块也都在，CI 也不一定会报错。我最开始是在自己单独建的虚拟环境里安装 wheel，然后发现 VS Code 没有自动提示，很多类型和函数也找不到，才意识到有问题。

我当时查了不少资料，一开始都没定位到原因。最后是从 VS Code 左侧文件树里看出端倪的：我发现 `_lib.pyi` 的文件名是灰色的，这才突然想到，会不会和 `.gitignore` 有关。因为被 ignore 的文件通常就会显示成灰色。

打开 `.gitignore` 一看，里面果然有一条：

```gitignore
*.pyi
```

删掉之后，一切正常。

这个问题给我印象很深，因为它非常典型：**打包问题不一定出在构建脚本里，反而可能出在某个很容易被忽略的配置文件里。**

后来我还试着去搜了一下 `cibuildwheel .gitignore`，结果只搜到一个 GitHub issue，讨论的大意也是是否能通过 `.gitignore` 来影响 wheel 打包内容。那个 issue 本身并没有给出特别明确的结论，但从我的实际情况来看，这类行为显然是会影响最终产物的。

---

### 2. GitHub Actions 的跨架构构建

第二个比较现实的问题是 CI 平台本身的限制。

GitHub 提供的免费 macOS runner 目前主要是 ARM 架构，但它又支持构建 ARM 和 x86_64 的 wheel。表面上看起来，好像两种架构都覆盖到了，但实际情况并没有这么理想。

在我的项目里，macOS x86_64 wheel 的构建是可以成功的，但测试阶段会失败。具体报错我当时没有完整记录下来，不过最终的结论很明确：虽然在 ARM 环境下可以通过虚拟化之类的方式构建出 x86_64 wheel，但一旦进入完整测试流程，某些依赖或者运行时环境就可能出问题。

最后的处理方式比较务实：**CI 里保留构建，但跳过 macOS x86_64 的自动测试。**

这当然不是完美方案，但在当前基础设施条件下，这是一个可以接受的折中。

除此之外，我还踩到过另一个跨架构构建的问题。最开始 Linux 的 `aarch64` 和 `x86_64` 我也是放在同一个镜像里编译的，似乎是以 `x86_64` 为主的环境。结果一到 `aarch64` 的构建阶段，速度就明显慢得离谱。后来我意识到，这本质上也是虚拟化带来的性能损耗。

再后来我把两个架构拆开，各自在对应的镜像里构建，`aarch64` 的速度提升非常明显，差不多有 10 倍左右。

这类问题很说明一件事：**“能构建” 和 “适合这样构建” 是两回事。**

---

### 3. `manylinux_2_28`：本地没感觉，老机器直接装不上

Linux wheel 的兼容性也是一个非常典型的问题：你在自己环境里完全感受不到，但用户一安装就会出问题。

一开始我用 `cibuildwheel` 的默认设置，构建出来的 Linux wheel 标签是 `manylinux_2_28`。当时我并没有太在意，直到后来才发现，旧环境根本装不上，比如 CentOS 7。

我是在 CentOS 7 上执行：

```bash
pip install traj-dist-rs
```

然后发现 `pip` 居然没有下载 wheel，而是下了一个 sdist 源码包，接着开始现场编译。我当时第一反应是很奇怪：为什么不装 wheel？

后来才明白，问题不在 `pip`，而在于这个系统太老，不满足 `manylinux_2_28` wheel 的兼容要求，所以 `pip` 只能退回到源码构建。

最后解决办法也很明确：显式改成 `manylinux2014`。

```yaml
CIBW_MANYLINUX_AARCH64_IMAGE: manylinux2014
CIBW_MANYLINUX_X86_64_IMAGE: manylinux2014
```

改完之后，wheel 的兼容范围就覆盖到了老系统，同时新系统也照样能用。还好当时我还处在 `1.0.0-rc.1` 的测试阶段，正式版还没发出去，不然后面修起来就更麻烦了。

这件事再次提醒我：**CI 构建成功，不等于用户环境里就能顺利安装。**

---

## 最后

现在再回头看，`traj-dist-rs` 这个项目带给我的收获，已经远远不只是“实现了一套轨迹距离算法”。

在代码层面：

- Rust 的 feature 非常适合拆分核心逻辑和 Python 绑定
- 零拷贝接口的关键往往不在“技巧”，而在于是否真正理解底层数据布局
- 并行库的默认行为在分布式环境里未必可靠

在工具链层面：

- MkDocs + Mike 非常适合中小型项目
- notebook 驱动的文档维护方式，在双语言项目里比我预期中更实用
- wheel 打包、CI 平台、manylinux 兼容性，这些看似琐碎，但都是真实的工程经验

---

## 补充：关于我现阶段对 AI Coding 的一些判断

最后这部分算是一个延伸。

`traj-dist-rs` 这个项目本身是我通过 AI Coding 辅助完成的，使用的是 iFlow CLI 和 GLM 4.7。整个开发过程总体上还是比较顺利的，但也正因为如此，我反而更清楚地感受到了现阶段 AI Coding 的能力边界。

先说说不顺利的地方：当遇到一些非常困难的疑难问题的时候，LLM由于基础模型能力的限制，并不能很好的发现并解决问题，此时可能会被自身幻觉导向“欺骗”用户，可能的结果是LLM说问题已经解决了，但是用户进去细看才会发现问题没有被正确解决。当然这实际不是欺骗，只是LLM的能力受限导致理解上有问题。

我现在一个很直接的判断是：**AI Coding 依然非常依赖 prompt 的质量。**

对于同一个软件，不同的人写出来的实现可能完全不同，不同的人对内部架构的理解也会非常不一样。因此，不同层次的 prompt 实际上会直接影响软件最终的结构。prompt 越清晰、越深入，AI 模型越容易给出更合理的实现。

而软件架构这件事，本身就会决定后期的维护成本、性能上限和可扩展性。

这也意味着：对于初级工程师来说，如果本身还缺乏比较成熟的软件架构经验，那么一方面很难稳定地引导 AI 写出高质量实现，另一方面就算 AI 生成了一个“看起来不错”的结果，自己也还需要花时间去理解和确认它到底是不是合理的。

从这个角度看，高级工程师的价值反而会被进一步放大。因为他们不仅能判断实现对不对，还能更早地影响系统应该怎么被设计。因此在未来几年里，我个人确实认为 AI Coding 会对初级工程师造成一定冲击。

不过换个角度看，这里面也有非常积极的一面。

如果我们要做的是一个边界比较清晰的产品，能够明确地定义它的输入、输出和交互形式，那么 AI 就很适合快速把它实现出来。对于这类需求，AI Coding 已经显著降低了原型开发和简单产品搭建的门槛。

也就是说，AI 一方面会提高工程门槛中“架构判断”的重要性，另一方面又会显著降低“把一个想法快速做出来”的成本。

所以我现在的感受是：

- 未来 1 到 2 年，企业对初级工程师的需求可能会下降
- 高级工程师的价值未必会下降，甚至可能进一步放大
- 基于 AI Coding 的创业赛道会更拥挤
- 年轻人把自己的 idea 做成产品的成本，会明显变低

这可能也是接下来几年里，软件开发领域最值得持续观察的一件事。

我之所以写下这篇博客，一个很核心的原因，其实还是想把自己在这个阶段学到的东西认真记录下来。

即便未来 AI Coding 真的进一步席卷整个行业，影响到各个工种，我想我大概还是会喜欢编程这件事。
可能我这辈子都不会讨厌编程吧。
