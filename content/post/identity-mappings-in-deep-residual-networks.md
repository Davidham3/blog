---
categories:
- 论文阅读笔记
date: 2018-03-08 18:45:45+0000
draft: false
math: true
tags:
- deep learning
- machine learning
- ResNet
- 已复现
title: Identity Mappings in Deep Residual Networks
---
ECCV 2016, ResNet v2, 原文链接：[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
<!--more-->

# Identity Mappings in Deep Residual Networks
## Introduction
Deep residual network (ResNets) consist of many stacked "Residual Units". Each unit (Fig. 1(a)) can be expressed in a general form:
$$y\_l = h(x\_l) + \mathcal{F}(x\_l, \mathcal{W\_l})$$
$$x\_{l+1}=f(y\_l)$$
where $x\_l$ and $x\_{l+1}$ are input and output of the $l$-th unit, and $\mathcal{F}$ is a residual function.$h(x\_l)=x\_l$ is an identity mapping and $f$ is a ReLU function.
The central idea of ResNets is to learn the additive residual function $\mathcal{F}$ with respect to $h(x\_l)$, with a key choice of using an identity mapping $h(x\_l)=x\_l$. This is realized by attaching an identity skip connection ("shortcut").
In this paper, we analyze deep residual networks by focusing on creating a "direct" path for propagating information -- not only within a residual unit, but through the entire network. Our derivations reveal that *if both h(x_l) and f(y_l) are identity mappings, the signal could be directly* propagated from one unit to any other units, in both forward and backward passes.
To understand the role of skip connections, we analyse and compare various types of $h(x\_l)$. We find that the identity mapping $h(x\_l) = x\_l$ chosen in achieves the fastest error reduction and lowest training loss among all variants we investigated, whereas skip connections of scaling, gating, and $1 \times 1$ convolutions all lead to higher training loss and error. These experiments suggest that keeping a "clean" information path (indicated by the grey arrows in Fig. 1,2, and 4) is helpful for easing optimization.
![Fig1](/images/identity-mappings-in-deep-residual-networks/Fig1.PNG)
Figure 1. Left: (a) original Residual Unit in [1]; (b) proposed Residual Unit. The grey arrows indicate the easiest paths for the information to propagate, corresponding to the additive term "x_l" in Eqn.(4) (forward propagation) and the additive term "1" in Eqn.(5) (backward propagation). Right: training curves on CIFAR-10 of 1001-layer ResNets. Solid lines denote test error (y-axis on the right), and dashed lines denote training loss (y-axis on the left). The proposed unit makes ResNet-1001 easier to train.

To construct an identity mapping $f(y\_l)=y\_l$, we view the activation functions (ReLU and BN) as "pre-activation" of the weight layers, in constrast to conventional wisdom of "post-activation". This point of view leads to a new residual unit design, shown in (Fig. 1(b)). Based on this unit, we present competitive results on CIFAR-10/100 with a 1001-layer ResNet, which is much easier to train and generalizes better than the original ResNet in [1]. We further report improved results on ImageNet using a 200-layer ResNet, for which the counterpart of [1] starts to overfit. These results suggest that there is much room to exploit the dimension of *network depth*, a key to the success of modern deep learning.

## Analysis of Deep Residual Networks
The ResNets developed in [1] are *modularized* architectures that stack building blocks of the same connecting shape. In this paper we call these blocks "Residual Units". The original Residual Unit in [1] performs the following computation:
$$y\_l = h(x\_l) + \mathcal{F}(x\_l, \mathcal{W-l})$$
$$x\_{l+1}=f(y\_l)$$
Here $x\_l$ is the input feature to the $l$-th Residual Unit. $\mathcal{W\_l}=\lbrace W\_{l,k} \mid 1 \le k \le K\rbrace$ is a set of weights (and biases) associated with the $l$-th Residual Unit, and $K$ is the number of layers in a Residual Unit ($K$ is 2 or 3 in [1]). $\mathcal{F}$ denotes the residual function, *e.g.*, a stack of two $3 \times 3$ convolutional layers in [1]. The function $f$ is the operation after element-wise addition, and in [1] $f$ is ReLU. The function $h$ is set as an identity mapping: $h(x\_l)=x\_l$.
If $f$ is also an identity mapping: $x\_{l+1} \equiv y\_l$, we can put Eqn.(2) into Eqn.(1) and obtain:
$$x\_{l+1}=x\_l+\mathcal{F}(x\_l, \mathcal{W\_l})$$
Recursively $(x\_{l+2}=x\_{l+1} + \mathcal{F}(x\_{l+1}, \mathcal{W\_{l+1}}) = x\_l + \mathcal{F}(x\_l, \mathcal{W\_l}) + \mathcal{F}(x\_{l+1},\mathcal{W\_{l+1}}), etc.)$ we will have:
$$x\_L = x\_l + \sum\_{i=1}^{L-1}\mathcal{F}(x\_i, \mathcal{W\_i})$$
for *any deeper unit* $L$ and *any shallower unit* $l$. Eqn.(4) exhibits some nice properties.
1. The feature $x\_L$ of any deeper unit $L$ can be represented as the feature $x\_l$ of any shallower unit $l$ plus a residual function in a form of $\sum\_{i=1}^{L-1}\mathcal{F}$, indicating that the model is in a *residual* fashion between any units $L$ and $l$.
2. The feature $x\_L = x\_0 + \sum\_{i=0}^{L-1}\mathcal{F}(x\_i, \mathcal{W\_i})$, of any deep unit $L$, is the *summation* of the outputs of all preceding residual functions (plus $x\_0$). This is in contrast to a "plain network" where a feature $x\_L$ is a series of matrix-vector *products*, say, $\prod\_{i=0}^{L-1}W\_ix\_0$ (ignoring BN and ReLU).
Eqn.(4) also leads to nice backward propagation properties. Denoting the loss function as $\varepsilon$, from the chain rule of backpropagation [9] we have:
$$\frac{\partial{\varepsilon}}{\partial{x\_l}}=\frac{\partial{\varepsilon}}{\partial{x\_L}}\frac{\partial{x\_L}}{\partial{x\_l}}=\frac{\partial{\varepsilon}}{\partial{x\_L}}(1+\frac{\partial}{\partial{x\_l}}\sum\_{i=l}^{L-1}\mathcal{F}(x\_i, \mathcal{W\_i}))$$
Eqn.(5) indicates that the gradient $\frac{\partial{\varepsilon}}{\partial{x\_i}}$ can be decomposed into two additive terms: a term of $\frac{\partial{\varepsilon}}{\partial{x\_L}}$ that propagates information directly without concerning any weight layers, and another term of $\frac{\partial{\varepsilon}}{\partial{x\_L}}(\frac{\partial}{\partial{x\_l}}\sum\_{i=l}^{L-1}\mathcal{F})$ that propagates through the weight layers. The additive term of $\frac{\partial{\varepsilon}}{\partial{x\_L}}$ ensures that information is directly propagated back to *any shallower unit* $l$. Eqn.(5) also suggests that it is unlikely for the gradient $\frac{\partial{\varepsilon}}{\partial{x\_l}}$ to be canceled out for a mini-batch, because in general the term $\frac{\partial}{\partial{x\_l}}\sum\_{i=l}^{L-1}\mathcal{F}$ cannot be always -1 for all samples in a mini-batch. This implies that the gradient of a layer does not vanish even when the weights are arbitrarily small.

## On the Importance of Identity Skip Connections
Let's consider a simple modification, $h(x\_l)=\lambda\_lx\_l$, to break the identity shortcut:
$$x\_{l+1}=\lambda\_lx\_l+\mathcal{F}(x\_l, \mathcal{W\_l})$$
where $\lambda\_l$ is a modulating scalar (for simplicity we still assume $f$ is identity).
Recursively applying this forumulation we obtain an equation similar to Eqn. (4): $x\_L=(\prod\_{i=l}^{L-1}\lambda\_i)x\_l+\sum\_{i=1}^{L-1}(\prod\_{j=i+1}^{L-1}\lambda\_j)\mathcal{F}(x\_i, \mathcal{W\_i})$, or simply:
$$x\_L = (\prod\_{i=l}^{L-1}\lambda\_i)x\_l+\sum\_{i=l}^{L-1}\hat{\mathcal{F}}(x\_i, \mathcal{W\_i})$$
where the notation $\hat{\mathcal{F}}$ absorbs the scalars into the residual functions. Similar to Eqn.(5), we have backpropagation of the following form:
$$\frac{\partial{\varepsilon}}{\partial{x\_l}}=\frac{\partial{\varepsilon}}{\partial{x\_L}}((\prod\_{i=l}^{L-1}\lambda\_i)+\frac{\partial}{\partial{x\_l}}\sum\_{i=l}^{L-1}\hat{\mathcal{F}}(x\_i, \mathcal{W\_i}))$$
For an extremely deep network ($L$ is large), if $\lambda\_i > 1$ for all $i$, this factor can be exponentially large; if $\lambda\_i < 1$ for all $i$, this factor can be expoentially small and vanish, which blocks the backpropagated signal from the shortcur and forces it to flow through the weighted layers. This results in optimization difficuties as we show by experiments.
If the skip connection $h(x\_l)$ represents more complicated transforms (such as gating and $1 \times 1$ convolutions), in Eqn.(8) the first term becomes $\prod\_{i=l}^{L-1}h\_i'$ where $h'$ is the derivative of $h$. This product may also impede information propagation and hamper the training procedure as witnessed in the following experiments.

### Experiments on skip Connections
We experiments with the 110-layer ResNet as presented in [1] on CIFAR-10. Though our above analysis is driven by identity $f$, the experiments in this section are all based on $f = ReLU$ as in [1]; we address identity $f$ in the next section. Our baseline ResNet-110 has 6.61% error on the test set. The comparisons of other variants (Fig.2 and Table 1) are summarized as follows:
**Table 1.** Classification error on the CIFAR-10 test set using ResNet-110 [1], with different types of shortcut connections applied to all Residual Units. We report "fail" when the test error is higher than 20%.
![Table1](/images/identity-mappings-in-deep-residual-networks/Table1.PNG)

**Constant scaling**. We set $\lambda = 0.5$ for all shortcuts (Fig. 2(b)). We further study two cases of scaling $\mathcal{F}$:
1. $\mathcal{F}$ is not scaled;
2. $\mathcal{F}$ is scaled by a constant scalar of $1-\lambda = 0.5$, which is similar to the highway gating [6,7] but with frozen gates. The former case does not converge well; the latter is able to converge, but the test error (Table 1, 12.35%) is substantially higher than the original ResNet-110. Fig 3(a) shows that the training error is higher than that of the original ResNet-110, suggesting that the optimization has difficulties when the shortcut signal is scaled down.

**Exclusive gating**. Following the Highway Networks [6,7] that adopt a gating mechanism [5], we consider a gating function $g(x)=\sigma(W\_gx+b\_g)$ where a transform is represented by weights $W\_g$ and biases $b\_g$ followed by the sigmoid function $\sigma(x)=\frac{1}{1+e^{-x}}$. In a convolutional network $g(x)$ is realized by a $1 \times 1$ convolutional layer. The gating function modulates the signal by element-wise multiplication.
We investigate the "exclusive" gates as used in [6,7] -- the $\mathcal{F}$ path is scaled by $g(x)$ and the shortcut path is scaled by $1-g(x)$. See Fig 2(c). We find that the initialization of the biases $b\_g$ is critical for training gated models, and following the guidelines in [6,7], we conduct hyper-parameter search on the initial value of $b\_g$ in the range of 0 to -10 with a decrement step of -1 on the training set by cross-validation. The best value (-6 here) is then used for training on the training set, leading to a test result of 8.70% (Table 1), which still lags far behind the ResNet-110 baseline. Fig 3(b) shows the training curves. Table 1 also reports the results of using other initialized values, noting that the exclusive gating network does not converge to a good solution when $b\_g$ is not appropriately initialized.

**Shortcut-only gating**. In this case the function $\mathcal{F}$ is not scaled; only the shortcut path is gated by $1-g(x)$. See Fig 2(d). The initialized value of $b\_g$ is still essential in this case. When the initialized $b\_g$ is 0 (so initially the expectation of $1-g(x)$ is 0.5), the network converges to a poor result of 12.86% (Table 1). This is also caused by higher training error (Fig 3(c)).
When the initialized $b\_g$ is very negatively biased (e.g., -6), the value of $1-g(x)$ is closer to 1 and the shortcut connection is nearly an identity mapping. Therefore, the result (6.91%, Table 1) is much closer to the ResNet-110 baseline.

**$1 \times 1$ convolutional shortcut**. Next we experiment with $1 \times 1$ convolutional shortcut connections that replace the identity. This option has been investigated in [1] (known as option C) on a 34-layer ResNet (16 Residual Units) and shows good results, suggesting that $1 \times 1$ shortcut connections could be useful. But we find that this is not the case when there are many Residual Units. The 110-layer ResNet has a poorer result (12.22%, Table 1) when using $1 \times 1$ convolutional shortcuts. Again, the training error becomes higher (Fig 3(d)). When stacking so many Residual Units (54 for ResNet-110), even the shortest path may still impede signal propagation. We witnessed similar phenomena on ImageNet with ResNet-101 when using $1 \times 1$ convolutional shortcuts.

**Dropout shortcut**. Last we experiment with dropout [11] (at a ratio of 0.5) which we adopt on the output of the identity shortcut (Fig. 2(f)). The network fails to converge to a good solution. Dropout statistically imposes a scale of $\lambda $ with an expectation of 0.5 on the shortcut, and similar to constant scaling by 0.5, it impedes signal propagation.

## On the Usage of Activation Functions
We want to make $f$ an identity mapping, which is done by re-arranging the activation function (ReLU and/or BN). The original Residual Unit in [1] has a shape in Fig.4(a) -- BN is used after each weight layer, and ReLU is adopted after BN expect that the last ReLU in a Residual Unit is after element-wise addition ($f=ReLU$). Fig.4(b-e) show the laternatives we investigated, explained as following.

## Experiments on Activation
In this section we experiment with ResNet-110 and a 164-layer Bottlenect [1] architecture (denoted as ResNet-164). A bottleneck Residual Unit consist of a $1 \times 1$ layer for reducing dimension, a $3 \times 3$ layer, and a $1 \times 1$ layer for restoring dimension. As designed in [1], its computational complexity is similar to the two-$3 \times 3$ Residual Unit. More details are in the appendix. The baseline ResNet-164 has a competitive result of 5.93% on CIFAR-10 (Table 2).

**BN after addition**. Before turning $f$ into an identity mapping, we go the opposite way by adopting BN after addition (Fig. 4(b)). In this case $f$ involves BN and ReLU. The results become considerably worse than the baseline (Table 2). Unlike the original design, now the BN layer alters the signal that passes through the shortcut and impedes information propagation, as reflected by the difficulties on reducing training loss at the begining of training (Fib. 6 left).

**ReLU before addition**. A naive choice of making $f$ into an identity mapping is to move the ReLU

## Implementation
使用mxnet实现了一版
```python
from mxnet import nd
import mxnet as mx
import numpy as np
import pickle
from mxnet import image
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dicts = pickle.load(fo, encoding='bytes')
    return dicts

def residual_unit(x, channels, name, same_shape = True):
    stride = 1 if same_shape else 2
    net = mx.sym.BatchNorm(data = x, fix_gamma = False, name = '%s_bn1'%(name), momentum=0.9)
    net = mx.sym.Activation(data = net, act_type = 'relu', name = '%s_relu1'%(name))
    net = mx.sym.Convolution(data = net, num_filter = channels, kernel = (3, 3),\
                             pad = (1, 1), stride = (stride, stride), name = '%s_conv1'%(name))
    net = mx.sym.BatchNorm(data = net, fix_gamma = False, name = '%s_bn2'%(name), momentum=0.9)
    net = mx.sym.Activation(data = net, act_type = 'relu', name = '%s_relu2'%(name))
    net = mx.sym.Convolution(data = net, num_filter = channels, kernel = (3, 3),\
                             pad = (1, 1), name = '%s_conv2'%(name))
    if not same_shape:
        x = mx.sym.Convolution(data = x, num_filter = channels, pad = (0, 0),\
                               stride = (stride, stride), kernel = (1, 1), name = "%s_conv3"%(name))
    return net + x

def ResNet(units, nums):
    data = mx.sym.Variable('data')
    net = mx.sym.Convolution(data, num_filter = 16, kernel = (3, 3), pad = (1, 1))
    
    for num in nums:
        net = residual_unit(net, num, 'r%s%s'%(num, 1), False)
        for i in range(2, units+1):
            net = residual_unit(net, num, 'r%s%s'%(num, i))
    
    net = mx.sym.BatchNorm(net, name = 'batch1', momentum=0.9)
    net = mx.sym.Activation(net, act_type = 'relu', name = 'relu1')
    net = mx.sym.Pooling(net, pool_type = 'avg', kernel = (3, 3), name = 'pool1')
    net = mx.sym.Flatten(net, name = 'flat1')
    net = mx.sym.FullyConnected(net, name = 'fc1', num_hidden = 10)
    net = mx.sym.SoftmaxOutput(net, name = 'softmax')
    return net

all_data = []
for i in range(1, 6):
    data = unpickle('../data/cifar-10-batches-py/data_batch_%s'%(i))
    all_data.append((nd.array(data[b'data']), nd.array(data[b'labels'])))
X, y = zip(*all_data)

trainX, trainY = nd.concat(*X, dim = 0).reshape(shape = (-1, 3, 32, 32)).astype('float32'),\
                    nd.concat(*y, dim = 0)
data = unpickle('../data/cifar-10-batches-py/test_batch')
testX = nd.array(data[b'data']).reshape(shape = (-1, 3, 32, 32)).astype('float32')
testY = nd.array(data[b'labels'])

# batch_size = 128
batch_size = 128
train_iter = mx.io.NDArrayIter(trainX, trainY, batch_size, shuffle = True)
test_iter = mx.io.NDArrayIter(testX, testY, batch_size, shuffle = False)

net = ResNet(12, [64, 128, 256])
mod = mx.mod.Module(symbol=net,
                    context=[mx.gpu(i) for i in range(0, 2)],
#                     context = mx.gpu(0),
                    data_names=['data'],
                    label_names=['softmax_label'])

mod.bind(data_shapes = train_iter.provide_data, label_shapes = train_iter.provide_label)
mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
mod.init_optimizer(optimizer='nag', optimizer_params=(('learning_rate', 0.1),
                                                       ('wd', 0.0001),
                                                       ('momentum', 0.9)))
# mod.init_optimizer(optimizer='adam', optimizer_params=(('learning_rate', 5e-4),
#                                                        ('beta1', 0.9),
#                                                        ('beta2', 0.99)))

losses = []
accuracy = []
metrics = [mx.metric.create('acc'), mx.metric.CrossEntropy()]
for epoch in range(10):
    train_iter.reset()
    [i.reset() for i in metrics]
    for batch in train_iter:
        mod.forward(batch, is_train = True)
        mod.update_metric(metrics[0], batch.label)
        mod.update_metric(metrics[1], batch.label)
        mod.backward()
        mod.update()
    if epoch % 1 == 0:
        score = mod.score(test_iter, ['acc', mx.metric.CrossEntropy()])
        losses.append((metrics[1].get()[-1], score[1][-1]))
        accuracy.append((metrics[0].get()[-1], score[0][-1]))
        print('Epoch %d, Training acc %s, loss %s'%(epoch, accuracy[-1][0], losses[-1][0]))
        print('Epoch %d, Validation acc %s, loss %s'%(epoch, accuracy[-1][1], losses[-1][1]))
        print()

plt.figure(figsize=(10, 8))
train_loss, test_loss = zip(*losses)
plt.plot(train_loss, '-', color = 'blue', label = 'training loss')
plt.plot(test_loss, '-', color = 'red', label = 'testing loss')
plt.legend(loc = 'upper right')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

plt.figure(figsize=(10, 8))
train_acc, test_acc = zip(*accuracy)
plt.plot(train_acc, '-', color = 'blue', label = 'training acc')
plt.plot(test_acc, '-', color = 'red', label = 'testing acc')
plt.legend(loc = 'upper right')
plt.xlabel('iteration')
plt.ylabel('acc')
plt.show()
```