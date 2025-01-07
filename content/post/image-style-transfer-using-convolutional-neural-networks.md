---
categories:
- 论文阅读笔记
date: 2018-02-24 23:51:39+0000
description: CVPR 2016，大体原理：选择两张图片，一张作为风格图片，一张作为内容图片，任务是将风格图片中的风格，迁移到内容图片上。方法也比较简单，利用在ImageNet上训练好的VGG19，因为这种深层次的卷积神经网络的卷积核可以有效的捕捉一些特征，越靠近输入的卷积层捕捉到的信息层次越低，而越靠近输出的卷积层捕捉到的信息层次越高，因此可以用高层次的卷积层捕捉到的信息作为对风格图片风格的捕捉。而低层次的卷积层用来捕捉内容图片中的内容。所以实际的操作就是，将内容图片扔到训练好的VGG19中，取出低层次的卷积层的输出，保存起来，然后再把风格图片放到VGG19中，取出高层次的卷积层的输出，保存起来。然后随机生成一张图片，扔到VGG19中，将刚才保存下来的卷积层的输出的那些卷积层的结果拿出来，和那些保存的结果做个loss，然后对输入的随机生成的图片进行优化即可。原文链接：[Image
  Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
draft: false
math: true
tags:
- deep learning
- machine learning
- computer vision
- image style transfer
- 已复现
title: Image Style Transfer Using Convolutional Neural Networks
---
CVPR 2016，大体原理：选择两张图片，一张作为风格图片，一张作为内容图片，任务是将风格图片中的风格，迁移到内容图片上。方法也比较简单，利用在ImageNet上训练好的VGG19，因为这种深层次的卷积神经网络的卷积核可以有效的捕捉一些特征，越靠近输入的卷积层捕捉到的信息层次越低，而越靠近输出的卷积层捕捉到的信息层次越高，因此可以用高层次的卷积层捕捉到的信息作为对风格图片风格的捕捉。而低层次的卷积层用来捕捉内容图片中的内容。所以实际的操作就是，将内容图片扔到训练好的VGG19中，取出低层次的卷积层的输出，保存起来，然后再把风格图片放到VGG19中，取出高层次的卷积层的输出，保存起来。然后随机生成一张图片，扔到VGG19中，将刚才保存下来的卷积层的输出的那些卷积层的结果拿出来，和那些保存的结果做个loss，然后对输入的随机生成的图片进行优化即可。原文链接：[Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
<!--more-->
# Image Style Transfer Using Convolutional Neural Networks
# 大体原理
选择两张图片，一张作为风格图片，一张作为内容图片，任务是将风格图片中的风格，迁移到内容图片上。方法也比较简单，利用在ImageNet上训练好的VGG19，因为这种深层次的卷积神经网络的卷积核可以有效的捕捉一些特征，越靠近输入的卷积层捕捉到的信息层次越低，而越靠近输出的卷积层捕捉到的信息层次越高，因此可以用高层次的卷积层捕捉到的信息作为对风格图片风格的捕捉。而低层次的卷积层用来捕捉内容图片中的内容。所以实际的操作就是，将内容图片扔到训练好的VGG19中，取出低层次的卷积层的输出，保存起来，然后再把风格图片放到VGG19中，取出高层次的卷积层的输出，保存起来。然后随机生成一张图片，扔到VGG19中，将刚才保存下来的卷积层的输出的那些卷积层的结果拿出来，和那些保存的结果做个loss，然后对输入的随机生成的图片进行优化即可。(Fig2)
![Fig2](/blog/images/image-style-transfer-using-convolutional-neural-networks/Fig2.PNG)

Figure 2. Style transfer algorithm. First content and style features are extracted and stored. The style image $\vec{a}$ is passed through the network and its style representation $A^l$ on all layers included are computed and stored(left). The content image $\vec{p}$ is passed through the network and the content representation $P^l$ in one layer is stored(right). Then a random white noise image $\vec{x}$ is passed through the network and its style features $G^l$ and content features $F^l$ are computed. On each layer included in the style representation, the element-wise mean squared difference between $G^l$ and $A^l$ is computed to give the style loss $\mathcal{L}\_{style}$(left). Also the mean squared difference between $F^l$ and $P^l$ is computed to give the content loss $\mathcal{L}\_{content}(right)$. The total loss $\mathcal{L}\_{total}$ is then a linear combination between the content and the style loss. Its derivative with respect to the pixel values can be computed using error back-propagation(middle). This gradient is used to iteratively update the image $\vec{x}$ until it simultaneously matches the style features of the style image $\vec{a}$ and the content features of the content image $\vec{p}$(middle, bottom).

# Deep image representations
We used the feature space provided by a normalized version of the 16 convolutional and 5 pooling layers of the 19-layer VGG network. We normalized the network by scaling the weights such that the mean activation of each convolutional filter over images and positions is equal to one. Such re-scaling can be done for the VGG network without changing its output, because it contains only rectifying linear activation functions and no normalization or pooling over feature maps.
其实这里我不是很明白为什么不会影响输出。

## content representation
A layer with $N\_l$ distinct filters has $N\_l$ feature maps each of size $M\_l$, where $M\_l$ is the height times the width of the feature map. So the responses in a layer $l$ can be stored in a matrix $F^l \in \mathcal{R}^{N\_l \times M\_l}$ where $F^l\_{ij}$ is the activation of the $i^{th}$ filter at position $j$ in layer $l$.
Let $\vec{p}$ and $\vec{x}$ be the original image and the image that is generated, and $P^l$ and $F^l$ their respective feature representation in layer $l$.
We then define the squared-error loss between the two feature representations
$$\mathcal{L}\_{content}(\vec{p}, \vec{x}, l) = \frac{1}{2}\sum\_{i, j}(F^l\_{ij}-P^l\_{ij})^2$$
The derivative of this loss with respect to the activations in layer $l$ equals

\begin{equation}
\frac{\partial{\mathcal{L}_{content}}}{\partial{F^l_{ij}}}=\left\{
\begin{aligned}
& (F^l - P^l)_{ij} & if \ F^l_{ij} > 0 \\
& 0 & if \ F^l_{ij} < 0
\end{aligned}
\right.
\end{equation}

from which the gradient with respect to the image $\vec{x}$ can be computed using standard error back-propagation.

When Convolutional Neural Networks are trained on object recongnition, they develop a representation of the image that makes object information increasingly explicit along the processing hierarchy. Higher layers in the network capture the high-level *content* in terms of objects and their arrangement in the input image but do not constrain the exact pixel values of the reconstruction very much. We therefore refer to the feature responses in higher layers of the network as the *content representation*.

## style representation
To obtain a representation of the style of an input image, we use a feature space designed to capture texture information. This feature space can be built on top of the filter responses in any layer of the network. It consists of the correlations between the different filter responses, where the expecation is taken over the spatial extent of the feature maps. These feature correlations are given by the Gram matrix $G^l \in \mathcal{R}^{N\_l \times N\_l}$, where $G^l\_{ij}$ is the inner product between the vecotrized feature maps $i$ and $j$ in layer $l$:
$$G^l\_{ij}=\sum\_kF^l\_{ik}F^l\_{jk}.$$
By inducing the feature corelations of multiple layers, we obtain a stationary, multi-scale representation of the input image, which captures its texture information but not the global arrangement. Again, we can visualise the information captured by these style feature spaces built on different layers of the network by constructing an image that matches the style representation of a given input image. This is done by using gradient descent from a white noise image to minimise the mean-squared distance between the entries of the Gram matrices from the original image and the Gram matrices of the image to be generated.
Let $\vec{a}$ and $\vec{x}$ be the original image and the image that is generated, and $A^l$ and $G^l$ their respective style representation in layer $l$. The contribution of layer $l$ to the toal loss is then
$$E\_l = \frac{1}{4N^2\_lM^2\_l}\sum\_{i,j}(G^l\_{ij} - A^l\_{ij})^2$$
and the total style loss is
$$\mathcal{L}\_{style}(\vec{a}, \vec{x})=\sum^L\_{l=0}w\_lE\_l,$$
where $w\_L$ are weighting factors of the contribution of each layer to the total loss (see below for specific values of $w\_l$ in our results). The derivative of $E\_l$ with respect to the activations in layer $l$ can be computed analytically:

\begin{equation}
\frac{\partial{E_l}}{\partial{F^l_{ij}}}=\left\{
\begin{aligned}
& \frac{1}{N^2_lM^2_l}((F^l)^T(G^l-A^l))_{ji} & if \ F^l_{ij} > 0 \\
& 0 & if \ F^l_{ij} < 0
\end{aligned}
\right.
\end{equation}
The gradient of $E\_l$ with respect to the pixel values $\vec{x}$ can be readily computed using standard error back-propagation.

## style transfer
To transfer the style of an artwork $\vec{a}$ onto a photograph $\vec{p}$ we synthesise a new image that simultaneously matches the content representation of $\vec{p}$ and the style representation of $\vec{a}$. Thus we jointly minimise the distance of the feature representations of a white noise image fron the content representation of the photograph in one layer and the style representation of the painting defined on a numebr of layers of the Convolutional Neural Network. The loss function we minimise is
$$\mathcal{L}\_{total}(\vec{p}, \vec{a}, \vec{x})=\alpha \mathcal{L}\_{content}(\vec{p}, \vec{x}) + \beta \mathcal{L}\_{style}(\vec{a}, \vec{x})$$
where $\alpha$ and $\beta$ are the weighting factors for content and style reconstruction, respectively. The gradient with respect to the pixel values $\frac{\partial{\mathcal{L}\_{total}}}{\partial{\vec{x}}}$ can be used as input for some numerical optimisation strategy. Here we use **L-BFGS**, which we found to work best for image synthesis. To extract image information on comparable scales, we always resized the style image to the same size as the content image before computing its feature representations.

# Results
## Trade-off between content and style matching
Since the loss function we minimise during image synthesis is a linear combination between the loss functions for content and style respectively, we can smoothly regulate the emphasis on either reconstructing the content or the style(Fig4).
![Fig4](/blog/images/image-style-transfer-using-convolutional-neural-networks/Fig4.PNG)
Figure 4. Relative weighting of matching content and style of the respective source images. The ratio $\alpha / \beta$ between matching the content and matching the style increases from top left to bottom right. A high emphasis on the style effectively produces a texturised version of the style image(top left). A high emphasis on the content produces an image with only little stylisation(bottom right). In practice one can smoothly interpolate between the two extremes.

## Effect of different layers of the Convolutional Neural Network
![Fig5](/blog/images/image-style-transfer-using-convolutional-neural-networks/Fig5.PNG)
Figure 5. The effect of matching the content representation in different layers of the network. Matching the content on layer 'conv2_2' preserves much of the fine structure of the original photograph and the synthesised image looks as if the texture of the painting is simply blended over the photograph(middle). When matching the content on layer 'conv4_2' the texture of the painting and the content of the photograph merge together such that the content of photograph is displayed in the style of the painting(bottom). Both images were generated with the same choice of parameters($\alpha / \beta = 1 \times 10^{-3}$). The painting that served as the style image is shown in the bottom left corner and is name <i>Jesuiten Ⅲ</i> by Lyonel Feininger, 1915.
Another important factor in the image synthesis process is the choice of layers to match the content and style representation on. As outlined above, the style representation is a multi-scale representation that includes multiple layers of the neural network. The number and position of these layers determines the local scale on which the style is matched, leading to different visual experiences. We find that matching the style representations up to higher layers in the network preserves local images structures an increasingly large scale, leading to a smoother and more continuous visual experience. Thus, the visually most appealing images are usually created by matching the style representation up to high layers in the network, which is why for all images shown we match the style features in layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1'and 'conv5_1' of the network.
To analyse the effect of using different layers to match the content features, we present a style transfer result obtained by stylising a photograph with the same artwork and parameter configuration ($\alpha / \beta = 1 \times 10^{-3}$), but in one matching the content features on layer 'conv2_2' and in the other on layer 'conv4_2'(Fig5). When matching the content on a lower layer of the network, the algorithm matches much of the detailed pixel information in the photograph and the generated image appears as if the texture of the artwork is merely blended over the photograph(Fig5, middle). In contrast, when matching the content features on a higher layer of the network, deatiled pixel information of the photograph is not as strongly constraint and the texture of the artwork and the content of the photograph are properly merged. That is, the fine structure of the image, for example the edges and colour map, is altered such that it agrees with the style of the artwork while displaying the content of the photograph(Fig5, bottom).

## Initialisation of gradient descent
![Fig6](/blog/images/image-style-transfer-using-convolutional-neural-networks/Fig6.PNG)
Figure 6. Initialisation of the gradient descent. <b>A</b> Initialised from the content image. <b>B</b> Initialised from the style image. <b>C</b> Four samples of images initialised from different white noise images. For all images the ratio $\alpha / \beta$ was equal to $1 \times 10^{-3}$
We have initialised all images shown so far with white noise. However, one could also initialise the image synthesis with either the content image or the style image. We explored these two alternatives(Fig6 A, B): although they bias the final image somewhat towards the spatial structure of the initialisation, the different intialisation do not seem to have a strong effect on the outcome of the synthesis procedure. It should be noted that only initialising with noise allows to generate an arbitrary number of new images(Fig6 C). Initialising with a fixed image always deterministically leads to the same outcome (up to stochasticity in the gradient descent procedure).

# implementation
关于实现的部分，我自己用mxnet实现了一下，但是发现和mxnet的example里面给的非常不一样。在他们的实现里面提到了Total variation denoising。而且，论文中的loss function是sum of square，而图2中给出是MSE，取了个平均值。我实现是时候没有取平均，导致loss很大，但是也可以训练。但是自己实现的梯度下降很难收敛，需要对梯度进行归一化，后来使用MXNet的gluon的Trainer训练会比原来好很多。
## Total variation denoising
In signal processing, total variation denoising, also known as total variation regularization, is a process, most often used in digital image processing, that has applications in noise removal.
!["Total variation denoising"](/blog/images/image-style-transfer-using-convolutional-neural-networks/ROF_Denoising_Example.png)
Example of application of the Rudin et al.[1] total variation denoising technique to an image corrupted by Gaussian noise. This example created using demo_tv.m by Guy Gilboa, see external links.
It is based on the principle that signals with excessive and possibly spurious detail have high total variation, that is, the integral of the absolute gradient of the signla is high. According to this principle, reducing the total variation of the signal subject to it being a close match to the original signal, removes unwanted detail whilst preserving important details such as edges. The concept was pioneered by Rudin, Osher, and Fatemi in 1992 and so is today known as the ROF model.
This noise removal technique has advantages over simple techniques such as linear smoothing or median filtering which reduce noise but at the same time smooth away edges to a greater or lesser degree. By contrast, total variation denoising is remarkably effective at simultaneously preserving edges whilst smoothing away noise in flat regions, even at low signal-to-noise ratios.
### 1D signal series
For a digital signal $y\_n$, we can, for example, define the total variation as:
$$V(y)=\sum\_n\vert y\_{n+1}-y\_n\vert$$
Given an input signal $x\_n$, the goal of total variation denoising is to find an approximation, call it $y\_n$, that has smaller total variation than $x\_n$ but is "close" to $x\_n$. One measure of closeness is the sum of square errors:
$$E(x, y)=\frac{1}{2}\sum\_n(x\_n - y\_n)^2$$
So the total variation denoising problem amounts to minimizing the following discrete functional over the signal $y\_n$:
$$E(x, y) + \lambda V(y)$$
By differentiating this functional with respect to $y\_n$, we can derive a corresponding Euler-lagrange equation, that can be numerically integrated with the original signal $x\_n$ as initial condition. This was the original approach. Alternatively, since this is a convex functional, techniques from convex optimization can be used to minimize it and find the solution $y\_n$.
### Regularization properties
The regularization parameter $\lambda $ plays a critical role in the denoising process. When $\lambda = 0$, there is no smoothing and the result is the same as minimizing the sum of squares. As $\lambda \to \infty $, however, the total variation term plays an increasingly strong role, which forces the result to have smaller total variation, at the expanse of being less like the input (noisy) signal. Thus, the choice of regularization parameter is critical to achieving just the right amount of noise removal.
### 2D signal images
We now consider 2D signals $y$, such as images. The total variation norm proposed by the 1992 paper is
$$V(y) = \sum\_{i,j}\sqrt{\vert y\_{i+1,j}-y\_{i,j}\vert ^2 + \vert y\_{i, j+1} - y\_{i, j}\vert ^2}$$
and is isotropic and not differentiable. A variation that is sometimes used, since it may sometimes be easier to minimize, is an anisotropic version
$$V\_{aniso}(y) = \sum\_{i,j}\sqrt{\vert y\_{i+1,j}-y\_{i,j}\vert ^2} + \sqrt{\vert y\_{i,j+1} - y\_{i,j}\vert ^2} = \sum\_{i,j}\vert y\_{i+1,j}-y\_{i,j}\vert + \vert y\_{i,j+1}-y\_{i,j}\vert $$
The standard total variation denoising problem is still of the form
$$\min\_yE(x,y)+\lambda V(y)$$
where $E$ is the 2D L2 norm. In contrast to the 1D case, solving this denoising is non-trivial. A recent algorithm that solves this is known as the primal dual method.
Due in part to much research in compressed sensing in the mid-2000s, there are many algorithms, such as the split-Bregman method, that solve variants of this problem.

不过我个人在实现的时候，实现了两个版本，一个是增加了total variation denoising，另一个是没增加total variation denoising的a。
代码如下：
```python
import mxnet as mx
from skimage import io
from skimage import transform
from mxnet import nd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon import nn
from mxnet import autograd
from mxnet.gluon import Trainer
from mxnet.gluon import Parameter

import warnings
warnings.filterwarnings("ignore")

content_image_path = '../../gluon-tutorial-zh/img/pine-tree.jpg'
style_image_path = 'the_starry_night.jpg'

rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def preprocessing(img, image_shape, ctx = mx.cpu()):
    newImage = transform.resize(img, image_shape)
    newImage = newImage.transpose((2, 0, 1))
    newImage = (newImage - rgb_mean.reshape(3, 1, 1)) / rgb_std.reshape(3, 1, 1)
    return nd.array(np.expand_dims(newImage, 0), ctx = ctx)

def postprocessing(img):
    newImage = img[0].asnumpy() * rgb_std.reshape(3, 1, 1) + rgb_mean.reshape(3, 1, 1)
    return newImage.transpose((1, 2, 0)).clip(0, 1)

def get_net(style_layers, content_layers):
    net = nn.HybridSequential()
    for i in range(max(style_layers + content_layers) + 1):
        net.add(pretrained_net.features[i])
    net.hybridize()
    return net

def extract_features(net, img, content_layers, style_layers):
    x = img.copy()
    content_results = []
    style_results = []
    for i in range(len(net)):
        x = net[i](x)
        if i in content_layers:
            content_results.append(x)
        if i in style_layers:
            style_results.append(x)
    return content_results, style_results

def content_loss(content_results, content_target):
    losses = []
    for i in range(len(content_results)):
        losses.append((content_results[i] - content_target[i]).square().sum())
    return nd.add_n(*losses) / 2

def gram(feature_map):
    N = feature_map.shape[1]
    M = np.prod(feature_map.shape[2:])
    new_feature_map = feature_map.reshape((N, M))
    return nd.dot(new_feature_map, new_feature_map.T)

def style_loss(style_results, style_target, weights):
    losses = []
    for i in range(len(style_results)):
        l = (gram(style_results[i]) - style_target[i]).square().sum() \
            / (4 * np.prod(style_results[i].shape[1:]))
        losses.append(weights[i] * l)
    return nd.add_n(*losses)

def get_loss(content_loss_result, style_loss_result, ratio):
    return content_loss_result * ratio + style_loss_result

style_layers = [2, 7, 16, 25, 34] # 这里与论文不同，我选的层比论文给出的更深，为了捕捉到更抽象的style
content_layers = [21]
net = get_net(style_layers, content_layers)

content_image = io.imread(content_image_path)
style_image = io.imread(style_image_path)

pretrained_net = models.vgg19(pretrained=True)

ctx = mx.gpu(1)
net.collect_params().reset_ctx(ctx)

content_img = preprocessing(content_image, (200, 300), ctx = ctx)
style_img = preprocessing(style_image, (200, 300), ctx = ctx)

output = Parameter('output', shape=content_img.shape)
output.initialize(ctx=ctx)
# output.set_data(nd.random_normal(shape = content_img.shape).abs())
output.set_data(content_img)

content_img_result, _ = extract_features(net, content_img, content_layers, style_layers)
_, style_img_result = extract_features(net, style_img, content_layers, style_layers)

content_results, style_results = extract_features(net, output.data(), content_layers, style_layers)
style_target = [gram(i) for i in style_img_result]


trainer = Trainer([output], 'adam',
                            {'learning_rate': 0.01, 'beta1': 0.9, 'beta2': 0.99})

for epoch in range(3000):
    with autograd.record():
        content_results, style_results = extract_features(net, output.data(), content_layers, style_layers)
        loss = get_loss(content_loss(content_results, content_img_result),
                        style_loss(style_results, style_target, [0.2] * 5),
                        1e-4)
    loss.backward()
    if epoch % 100 == 0:
        print(loss.asscalar())
    trainer.step(1)

plt.imshow(postprocessing(output.data()))
```

这里在实现的时候，使用了这个2D图像的total variation denoising，也就是，每个像素应尽可能的与左侧和上方的像素相近。所以最后的优化目标是三部分组成，第一部分是content loss，第二部分是style loss，第三部分是total variation loss。
研究一下mxnet给出的example
model_vgg19.py
```python
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import find_mxnet
import mxnet as mx
import os, sys
from collections import namedtuple

ConvExecutor = namedtuple('ConvExecutor', ['executor', 'data', 'data_grad', 'style', 'content', 'arg_dict'])

def get_symbol():
    # declare symbol
    data = mx.sym.Variable("data")
    conv1_1 = mx.symbol.Convolution(name='conv1_1', data=data , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu1_1 = mx.symbol.Activation(name='relu1_1', data=conv1_1 , act_type='relu')
    conv1_2 = mx.symbol.Convolution(name='conv1_2', data=relu1_1 , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu1_2 = mx.symbol.Activation(name='relu1_2', data=conv1_2 , act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=relu1_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv2_1 = mx.symbol.Convolution(name='conv2_1', data=pool1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1 , act_type='relu')
    conv2_2 = mx.symbol.Convolution(name='conv2_2', data=relu2_1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2 , act_type='relu')
    pool2 = mx.symbol.Pooling(name='pool2', data=relu2_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv3_1 = mx.symbol.Convolution(name='conv3_1', data=pool2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1 , act_type='relu')
    conv3_2 = mx.symbol.Convolution(name='conv3_2', data=relu3_1 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_2 = mx.symbol.Activation(name='relu3_2', data=conv3_2 , act_type='relu')
    conv3_3 = mx.symbol.Convolution(name='conv3_3', data=relu3_2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_3 = mx.symbol.Activation(name='relu3_3', data=conv3_3 , act_type='relu')
    conv3_4 = mx.symbol.Convolution(name='conv3_4', data=relu3_3 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_4 = mx.symbol.Activation(name='relu3_4', data=conv3_4 , act_type='relu')
    pool3 = mx.symbol.Pooling(name='pool3', data=relu3_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv4_1 = mx.symbol.Convolution(name='conv4_1', data=pool3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu4_1 = mx.symbol.Activation(name='relu4_1', data=conv4_1 , act_type='relu')
    conv4_2 = mx.symbol.Convolution(name='conv4_2', data=relu4_1 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu4_2 = mx.symbol.Activation(name='relu4_2', data=conv4_2 , act_type='relu')
    conv4_3 = mx.symbol.Convolution(name='conv4_3', data=relu4_2 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu4_3 = mx.symbol.Activation(name='relu4_3', data=conv4_3 , act_type='relu')
    conv4_4 = mx.symbol.Convolution(name='conv4_4', data=relu4_3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu4_4 = mx.symbol.Activation(name='relu4_4', data=conv4_4 , act_type='relu')
    pool4 = mx.symbol.Pooling(name='pool4', data=relu4_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv5_1 = mx.symbol.Convolution(name='conv5_1', data=pool4 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu5_1 = mx.symbol.Activation(name='relu5_1', data=conv5_1 , act_type='relu')

    # style and content layers
    style = mx.sym.Group([relu1_1, relu2_1, relu3_1, relu4_1, relu5_1])
    content = mx.sym.Group([relu4_2])
    return style, content


def get_executor(style, content, input_size, ctx):
    out = mx.sym.Group([style, content])
    # make executor
    arg_shapes, output_shapes, aux_shapes = out.infer_shape(data=(1, 3, input_size[0], input_size[1]))
    arg_names = out.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes]))
    grad_dict = {"data": arg_dict["data"].copyto(ctx)}
    # init with pretrained weight
    pretrained = mx.nd.load("./model/vgg19.params")
    for name in arg_names:
        if name == "data":
            continue
        key = "arg:" + name
        if key in pretrained:
            pretrained[key].copyto(arg_dict[name])
        else:
            print("Skip argument %s" % name)
    executor = out.bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, grad_req="write")
    return ConvExecutor(executor=executor,
                        data=arg_dict["data"],
                        data_grad=grad_dict["data"],
                        style=executor.outputs[:-1],
                        content=executor.outputs[-1],
                        arg_dict=arg_dict)


def get_model(input_size, ctx):
    style, content = get_symbol()
    return get_executor(style, content, input_size, ctx)
```

nstyle.py
```python
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import find_mxnet
import mxnet as mx
import numpy as np
import importlib
import logging
logging.basicConfig(level=logging.DEBUG)
import argparse
from collections import namedtuple
from skimage import io, transform
from skimage.restoration import denoise_tv_chambolle

CallbackData = namedtuple('CallbackData', field_names=['eps','epoch','img','filename'])

def get_args(arglist=None):
    parser = argparse.ArgumentParser(description='neural style')

    # 选择模型，默认是VGG19
    parser.add_argument('--model', type=str, default='vgg19',
                        choices = ['vgg'],
                        help = 'the pretrained model to use')

    # 内容图片的路径
    parser.add_argument('--content-image', type=str, default='input/IMG_4343.jpg',
                        help='the content image')

    # 风格图片的路径
    parser.add_argument('--style-image', type=str, default='input/starry_night.jpg',
                        help='the style image')
    
    # 停止迭代的阈值，若relative change小于这个数就停止迭代
    parser.add_argument('--stop-eps', type=float, default=.005,
                        help='stop if the relative chanage is less than eps')
    
    # 内容图片在loss上的权重
    parser.add_argument('--content-weight', type=float, default=10,
                        help='the weight for the content image')

    # 风格图片在loss上的权重
    parser.add_argument('--style-weight', type=float, default=1,
                        help='the weight for the style image')
    
    # total variation在loss上的权重
    parser.add_argument('--tv-weight', type=float, default=1e-2,
                        help='the magtitute on TV loss')

    # 最大迭代次数
    parser.add_argument('--max-num-epochs', type=int, default=1000,
                        help='the maximal number of training epochs')

    # 
    parser.add_argument('--max-long-edge', type=int, default=600,
                        help='resize the content image')

    # 初始的学习率
    parser.add_argument('--lr', type=float, default=.001,
                        help='the initial learning rate')

    # 使用哪块GPU
    parser.add_argument('--gpu', type=int, default=0,
                        help='which gpu card to use, -1 means using cpu')

    # 输出图像的路径
    parser.add_argument('--output_dir', type=str, default='output/',
                        help='the output image')

    # 每多少轮保存一次当前的输出结果
    parser.add_argument('--save-epochs', type=int, default=50,
                        help='save the output every n epochs')

    # 
    parser.add_argument('--remove-noise', type=float, default=.02,
                        help='the magtitute to remove noise')

    # 每迭代多少轮减小一下学习率
    parser.add_argument('--lr-sched-delay', type=int, default=75,
                        help='how many epochs between decreasing learning rate')

    # 学习率衰减因子
    parser.add_argument('--lr-sched-factor', type=int, default=0.9,
                        help='factor to decrease learning rate on schedule')

    if arglist is None:
        return parser.parse_args()
    else:
        return parser.parse_args(arglist)


def PreprocessContentImage(path, long_edge):
    '''
    内容图片预处理
    Parameter: path, str, 图片路径
               long_edge, int, float, str(float), 图像被缩放后长边的长度
    '''
    # 读取图片，使用skimage.io.imread，返回numpy.ndarray
    img = io.imread(path)

    # img.shape前两个数分别是多少行和多少列，第三个数是channel数
    logging.info("load the content image, size = %s", img.shape[:2])

    # resize一下图片，resize后的范围在0到1内
    factor = float(long_edge) / max(img.shape[:2])
    new_size = (int(img.shape[0] * factor), int(img.shape[1] * factor))
    resized_img = transform.resize(img, new_size)

    # 乘以256恢复到原来的区间
    sample = np.asarray(resized_img) * 256


    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)

    # sub mean，这里的均值应该是ImageNet数据集在RGB三通道上的均值
    sample[0, :] -= 123.68
    sample[1, :] -= 116.779
    sample[2, :] -= 103.939

    logging.info("resize the content image to %s", new_size)
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))

def PreprocessStyleImage(path, shape):
    '''
    对风格图片的预处理
    Parameter: path, str, 图像路径
               shape, tuple, 长度为4的tuple，第三个元素和第四个元素是content image的size
    '''
    img = io.imread(path)
    resized_img = transform.resize(img, (shape[2], shape[3]))
    sample = np.asarray(resized_img) * 256
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)

    sample[0, :] -= 123.68
    sample[1, :] -= 116.779
    sample[2, :] -= 103.939
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))

def PostprocessImage(img):
    '''
    对图像的后处理
    Parameter: img, numpy.ndarray
    '''
    img = np.resize(img, (3, img.shape[2], img.shape[3]))
    img[0, :] += 123.68
    img[1, :] += 116.779
    img[2, :] += 103.939
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 0, 2)

    # clip函数是用来砍掉小于下界和大于上届的数的
    img = np.clip(img, 0, 255)
    return img.astype('uint8')

def SaveImage(img, filename, remove_noise=0.):
    '''
    保存图片
    Parameter: img, numpy.ndarray
               filename, str
               remove_noise, float, default=0., 
    '''
    logging.info('save output to %s', filename)
    out = PostprocessImage(img)
    if remove_noise != 0.0:
        out = denoise_tv_chambolle(out, weight=remove_noise, multichannel=True)
    io.imsave(filename, out)

def style_gram_symbol(input_size, style):
    '''
    Parameter: input_size, tuple, length=2, 表示content image的size
               style, mx.sym.Group，里面是style对应的层
    '''
    _, output_shapes, _ = style.infer_shape(data=(1, 3, input_size[0], input_size[1]))
    gram_list = []
    grad_scale = []
    for i in range(len(style.list_outputs())):
        shape = output_shapes[i]
        x = mx.sym.Reshape(style[i], target_shape=(int(shape[1]), int(np.prod(shape[2:]))))
        # use fully connected to quickly do dot(x, x^T)
        gram = mx.sym.FullyConnected(x, x, no_bias=True, num_hidden=shape[1])
        gram_list.append(gram)

        # grad_scale c*h*w*c
        grad_scale.append(np.prod(shape[1:]) * shape[1])
    return mx.sym.Group(gram_list), grad_scale


def get_loss(gram, content):
    gram_loss = []
    for i in range(len(gram.list_outputs())):
        gvar = mx.sym.Variable("target_gram_%d" % i)
        gram_loss.append(mx.sym.sum(mx.sym.square(gvar - gram[i])))
    cvar = mx.sym.Variable("target_content")
    content_loss = mx.sym.sum(mx.sym.square(cvar - content))
    return mx.sym.Group(gram_loss), content_loss

def get_tv_grad_executor(img, ctx, tv_weight):
    """create TV gradient executor with input binded on img
    """
    if tv_weight <= 0.0:
        return None
    nchannel = img.shape[1]
    simg = mx.sym.Variable("img")
    skernel = mx.sym.Variable("kernel")
    channels = mx.sym.SliceChannel(simg, num_outputs=nchannel)
    out = mx.sym.Concat(*[
        mx.sym.Convolution(data=channels[i], weight=skernel,
                           num_filter=1,
                           kernel=(3, 3), pad=(1,1),
                           no_bias=True, stride=(1,1))
        for i in range(nchannel)])
    kernel = mx.nd.array(np.array([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]])
                         .reshape((1, 1, 3, 3)),
                         ctx) / 8.0
    out = out * tv_weight
    return out.bind(ctx, args={"img": img,
                               "kernel": kernel})

def train_nstyle(args, callback=None):
    """Train a neural style network.
    Args are from argparse and control input, output, hyper-parameters.
    callback allows for display of training progress.
    """
    # input
    dev = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
    content_np = PreprocessContentImage(args.content_image, args.max_long_edge)
    style_np = PreprocessStyleImage(args.style_image, shape=content_np.shape)

    # size是内容图片的尺寸
    size = content_np.shape[2:]

    # model
    Executor = namedtuple('Executor', ['executor', 'data', 'data_grad'])

    # 导入'model_vgg19.py'
    model_module =  importlib.import_module('model_' + args.model)

    # 获取到style和content两个mx.sym.Group，里面装着style和content层
    style, content = model_module.get_symbol()

    # 获取到所有style层的gram矩阵和grad scale
    gram, gscale = style_gram_symbol(size, style)

    
    model_executor = model_module.get_executor(gram, content, size, dev)
    model_executor.data[:] = style_np
    model_executor.executor.forward()
    style_array = []
    for i in range(len(model_executor.style)):
        style_array.append(model_executor.style[i].copyto(mx.cpu()))

    model_executor.data[:] = content_np
    model_executor.executor.forward()
    content_array = model_executor.content.copyto(mx.cpu())

    # delete the executor
    del model_executor

    style_loss, content_loss = get_loss(gram, content)
    model_executor = model_module.get_executor(
        style_loss, content_loss, size, dev)

    grad_array = []
    for i in range(len(style_array)):
        style_array[i].copyto(model_executor.arg_dict["target_gram_%d" % i])
        grad_array.append(mx.nd.ones((1,), dev) * (float(args.style_weight) / gscale[i]))
    grad_array.append(mx.nd.ones((1,), dev) * (float(args.content_weight)))

    print([x.asscalar() for x in grad_array])
    content_array.copyto(model_executor.arg_dict["target_content"])

    # train
    # initialize img with random noise
    img = mx.nd.zeros(content_np.shape, ctx=dev)
    img[:] = mx.rnd.uniform(-0.1, 0.1, img.shape)

    lr = mx.lr_scheduler.FactorScheduler(step=args.lr_sched_delay,
            factor=args.lr_sched_factor)

    optimizer = mx.optimizer.NAG(
        learning_rate = args.lr,
        wd = 0.0001,
        momentum=0.95,
        lr_scheduler = lr)
    optim_state = optimizer.create_state(0, img)

    logging.info('start training arguments %s', args)
    old_img = img.copyto(dev)
    clip_norm = 1 * np.prod(img.shape)
    tv_grad_executor = get_tv_grad_executor(img, dev, args.tv_weight)

    for e in range(args.max_num_epochs):
        img.copyto(model_executor.data)
        model_executor.executor.forward()
        model_executor.executor.backward(grad_array)
        gnorm = mx.nd.norm(model_executor.data_grad).asscalar()
        if gnorm > clip_norm:
            model_executor.data_grad[:] *= clip_norm / gnorm

        if tv_grad_executor is not None:
            tv_grad_executor.forward()
            optimizer.update(0, img,
                             model_executor.data_grad + tv_grad_executor.outputs[0],
                             optim_state)
        else:
            optimizer.update(0, img, model_executor.data_grad, optim_state)
        new_img = img
        eps = (mx.nd.norm(old_img - new_img) / mx.nd.norm(new_img)).asscalar()

        old_img = new_img.copyto(dev)
        logging.info('epoch %d, relative change %f', e, eps)
        if eps < args.stop_eps:
            logging.info('eps < args.stop_eps, training finished')
            break

        if callback:
            cbdata = {
                'eps': eps,
                'epoch': e+1,
            }
        if (e+1) % args.save_epochs == 0:
            outfn = args.output_dir + 'e_'+str(e+1)+'.jpg'
            npimg = new_img.asnumpy()
            SaveImage(npimg, outfn, args.remove_noise)
            if callback:
                cbdata['filename'] = outfn
                cbdata['img'] = npimg
        if callback:
            callback(cbdata)

    final_fn = args.output_dir + '/final.jpg'
    SaveImage(new_img.asnumpy(), final_fn)


if __name__ == "__main__":
    args = get_args()
    train_nstyle(args)
```