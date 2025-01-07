---
categories:
- 论文阅读笔记
date: 2018-03-04 18:59:20+0000
description: CVPR 2015，ResNet，原文链接：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
draft: false
math: true
tags:
- deep learning
- machine learning
- ResNet
title: Deep Residual Learning for Image Recognition
---
CVPR 2015，ResNet，原文链接：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
<!--more-->

# Deep Residual Learning for Image Recongnition
## problems
When deeper networks are able to start converging, a degradation problem has been exposed: with the network depth increasing, accuracy gets saturated (which might be unsurprising) and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error, as reported in [11, 42] and thoroughly verified by our experiments. Fig. 1 shows a typical example.
![Fig1](/blog/images/deep-residual-learning-for-image-recognition/Fig1.PNG)

Figure 1. Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer “plain” networks. The deeper network has higher training error, and thus test error. Similar phenomena on ImageNet is presented in Fig. 4.

The degradation (of training accuracy) indicates that not all systems are similarly easy to optimize. Let us consider a shallower architecture and its deeper counterpart that adds more layers onto it. There exists a solution *by construction* to the deeper model: the added layers are *identity* mapping and the other layers are copied from the learned shallower model. The existence of this constructed solution indicates that a deeper model should produce no higher training error than its shallower counterpart. But experiments show that our current solvers on hand are unable to find solutions that are comparably good or better than the constructed solution (or unable to do so in feasible time).

## Deep Residual Learning
### Residual Learning
Let us consider $\mathcal{H}(x)$ as an underlying mapping to be fit by a few stacked layers (not necessarily the entire net), with $x$ denoting the inputs to the first of these layers. If one hypothesizes that multiple nonlinear layers can asymptotically approximate complicated functions, then it is equivalent to hypothesize that they can asymptotically approximate the residual functions, $i.e.$, $\mathcal{H}(x)-x$ (assuming that the input and output are of the same dimensions). So rather than expect stacked layers to approximate $\mathcal{H}(x)$, we explicitly let these layers approximate a residual function $\mathcal{F}(x):=\mathcal{H}(x)-x$. The original function thus becomes $\mathcal{F}(x)+x$. Although both forms should be able to asymptotically approximate the desired functions (as hypothesized), the ease of learning might be different.

### Identity Mapping by Shortcuts
$$y = \mathcal{F}(x, {W\_i})+x$$
Here $x$ and $y$ are the input and output vectors of the layers considered. The function $\mathcal{F}(x, W\_i)$ represents the residual mapping to be learned. For the example in Fig. 2 that has two layers, $\mathcal{F} = W\_2\sigma (W\_1x)$ in which $\sigma $ denotes ReLU and the bias are omitting for simplifying notations. The operation $\mathcal{F}+x$ is performed by a shortcut connection and element-wise addition. We adopt the second nonlinearity after the addtion (*i.e.*, $\sigma(y)$, see Fig.2).
![Fig2](/blog/images/deep-residual-learning-for-image-recognition/Fig2.PNG)

Figure2. Residual learning: a building block.

The dimensions of $x$ and $\mathcal{F}$ must be equal in Eqn.(1). If this is not the case (*e.g.*, when changing the input/output channels), we can perform a linear projection $W\_s$ by the shortcut connections to match the dimensions:
$$y = \mathcal{F}(x, {W\_i}) + W\_sx$$
We can also use a square matrix $W\_s$ in Eqn.(1). But we will show by experiments that the identity mapping is sufficient for addressing the degradation problem and is economical, and thus $W\_s$ is only used when matching dimensions.
We also note that although the above notations are about fully-connected layers for simplicity, they are applicable to convolutional layers. The function $\mathcal{F}(x, {W\_i})$ can represent multiple convolutional layers. The element-wise addition is performed on two feature maps, channel by channel.

## Residual Network
The identity shortcuts can be directly used when the input and output are of the same dimensions (solid line shortcuts in Fig.3). When the dimensions increase (dotted line shortcuts in Fig.3), we consider two options: (A) The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions. This option introduces no extra parameter; (B) The projection shortcut in Eqn.(2) is used to match dimensions (done by $1 \times 1$ convolutions). For both options, when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2.
![Fig3](/blog/images/deep-residual-learning-for-image-recognition/Fig3.PNG)

Figure3. Example network architectures for ImageNet. <b>Left</b>: the VGG-19 model. <b>Middle</b>: a plain network with 34-parameter layers. **Right**: a residual network with 34 parameter layers. The dotted shortcuts increase dimensions. <b>Table 1</b> shows more details and other variants.

## Implementation
Our implementation for ImageNet follows the practice in *[Imagenet classification
with deep convolutional neural networks]* and *[Very deep convolutional networks for large-scale image recognition]*. 
1. The Image is resized with its shorter side randomly sampled in $[256, 480]$ for scale agumentation.
2. A $224 \times 224$ crop is randomly sampled from an image or its horizontal flip, with the per-pixel mean subtracted.
3. The standard color augmentation in *Imagenet classification
with deep convolutional neural networks* is used.
4. We adopt batch normalization (BN) right after each convolution and before activation.
5. We initialize the weights as in *Delving deep into rectifiers:
Surpassing human-level performance on imagenet classification* and train all plain/residual nets from scratch.
6. We use SGD with a mini-batch size of 256.
7. The learning rate starts from 0.1 and is divided by 10 when the error plateaus, and the models are trained from up to $60 \times 10^4$ iterations.
8. We use a weight decay of 0.0001 and a momentum of 0.9.
9. We do not use dropout, following the practice in *Batch normalization: Accelerating deep
network training by reducing internal covariate shift*.
10. In testing, for comparison studies we adopt the standard 10-crop testing.[*Imagenet classification
with deep convolutional neural networks*]
11. For best results, we adopt the fully-convolutional form as in *Very deep convolutional networks for large-scale image recognition* and *Delving deep into rectifiers: Surpassing human-level performance on imagenet classification*, and average the scores at multiple scales (images are resized such that the shorter side is in $\lbrace 224, 256, 384, 480, 640\rbrace $.

## ImageNet classification
### Deeper Bottleneck Architecture
Next we describe our deeper nets for ImageNet. Because of concerns on the training time that we can afford, we modify the building block as a *bottleneck* design. For each residual function $\mathcal{F}$, we use a stack of 3 layers instead of 2 (Fig. 5). The three layers are $1 \times 1$, $3 \times 3$, and $1 \times 1$ convolutions, where the $1 \times 1$ layers are responsible for reducing and then increasing (restoring) dimensions, leaving the $3 \times 3$ layer a bottleneck with smaller input/output dimensions. Fig. 5 shows an example, where both designs have similar time complexity.
The parameter-free indentity shortcuts are particularly important for the bottleneck architectures. If the identity

## CIFAR-10 and Analysis
The plain/residual architectures follow the form in Fig.3(middle/right). The network inputs are $32 \times 32$ images, with the per-pixel mean subtracted. The first layer is $3 \times 3$ convolutions. Then we use a stack of $6n$ layers with $3 \times 3$ convolutions on the feature maps of sizes $\lbrace 32, 16, 8\rbrace $ respectively, with $2n$ layers for each feature map size. The numbers of filters are $\lbrace 16, 32, 64\rbrace $ respectively, with $2n$ layers for each feature map size. The subsampling is performed by convolutions with a stride of 2. The network ends with a global average pooling, a 10-way fully-connected layer, and softmax.
When shortcut connections are used, they are connected to the pairs of $3 \times 3$ layers(totally $3n$ shortcuts). On this dataset we use identity shortcuts in all cases (*i.e.*, option A), so our residual models have exactly the same depth, width and number of parameters as the plain counterparts.
We use a weight decay of 0.0001 and momentum of 0.9, and adopt the weight initialization in *Delving deep into rectifiers: Surpassing human-level performance on imagenet classification* and BN in *Accelerating deep network training by reducing internal covariate shift* but with no dropout. These models are trained with a mini-batch size of 128 on two GPUs. We start with a learning rate of 0.1, divide it by 10 at 32k and 48k iterations, and terminate training at 64k iterations, which is determined on a 45k/5k train/val split. We follow the simple data augmentation in *Deeply-supervised nets* for training: 4 pixels are padded on each side, and a $32 \times 32$ crop is randomly sampled from the padded image or its horizontal flip. For testing, we only evaluate the single view of the original $32 \times 32$ image.
We compare $n=\lbrace 3, 5, 7, 9\rbrace $, leading to 20, 32, 44, and 56-layer networks.