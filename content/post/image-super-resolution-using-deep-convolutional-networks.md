---
categories:
- 论文阅读笔记
date: 2018-03-02 11:19:50+0000
description: PAMI 2016，大体思路：把训练集中的所有样本模糊化，扔到三层的卷积神经网络中，把输出和原始图片做一个loss，训练模型即可。原文链接：[Image
  Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)
draft: false
math: true
tags:
- deep learning
- machine learning
- super resolution
title: Image Super-Resolution Using Deep Convolutional Networks
---
PAMI 2016，大体思路：把训练集中的所有样本模糊化，扔到三层的卷积神经网络中，把输出和原始图片做一个loss，训练模型即可。原文链接：[Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)
<!--more-->

首先是ill-posed problem，图像的不适定问题
法国数学家阿达马早在19世纪就提出了不适定问题的概念:称一个数学物理定解问题的解存在、唯一并且稳定的则称该问题是适定的（Well Posed）.如果不满足适定性概念中的上述判据中的一条或几条，称该问题是不适定的。

# Convolutional Neural Networks For Super-Resolution
## Formulation
We first upscale a single low-resolution image to the desired size using bicubic interpolation. Let us denote the interpolated image as $Y$. Our goal is to recover from $Y$ an image $F(Y)$ that is as similar as possible to the ground truth high-resolution image $X$. For the ease of presentation, we still call $Y$ a "low-resolution" image, although it has the same size as $X$. We wish to learning a mapping $F$, which conceptually consists of three operations:
1. Patch extraction and representation. this operation extracts (overlapping) patches from the low-resolution image $Y$ and represents each patch as a high-dimensional vector. These vectors comprise a set of feature maps, of which the number equals to the dimensionality of the vectors.
2. Non-linear mapping. this operation nonlinearly maps each high-dimensional vector onto another high-dimensional vector. Each mapped vector is conceptually the representation of a high-resolution patch. These vectors comprise another set of feature maps.
3. Reconstruction. this operation aggregates the above high-resolution patch-wise representations to generate the final high-resolution image. This image is expected to be similar to the ground truth $X$.

### Patch Extraction and Representation
A popular strategy in image restoration is to densely extract patches and then represent them by a set of pre-trained bases such as PCA, DCT, Haar, etc. This is equivalent to convolving the image by a set of filters, each of which is a basis. In our formulation, we involve the optimization of these bases into the optimization of the network. Formally, our first layer is expressed as an operation $F\_1$:
$$F\_1(Y)=max(0, W\_1 * Y + B\_1)$$
where $W\_1$ and $B\_1$ represent the filters and biases respectively, and $\*$ denotes the convolution operation. Here, $W\_1$ corresponds to $n\_1$ filters of support $c \times f\_1 \times f\_1$, where $c$ is the number of channels in the input image, $f\_1$ is the spatial size of a filter. Intuitively, $W\_1$ applies $n\_1$ convolutions on the image, and each convolution has a kernel size $c \times f\_1 \times f\_1$. The output is composed of $n\_1$ feature maps. $B\_1$ is an $n\_1$-dimensional vector, whose each element is associated with a filter. We apply the ReLU on the filter responses.

### Non-Linear Mapping
The first layer extracts an $n\_1$-dimensional feature for each patch. In the second operation, we map each of these $n\_1$-dimensional vectors into an $n\_2$-dimensional one. This is equivalent to applying $n\_2$ filters which have a trivial spatial support $1 \times 1$. This interpretation is only valid for $1 \times 1$ filters. But it is easy to generalize to larger filters like $3 \times 3$ or $5 \times 5$. In that case, the non-linear mapping is not on a patch of the input image; instead, it is on a $3 \times 3$ or $5 \times 5$ "patch" of the feature map. The operation of the second layer is:
$$F\_2(Y) = max(0, W\_2 * F\_1(Y) + B\_2)$$
Here $W\_2$ contains $n\_2$ filters of size $n\_1 \times f\_2 \times f\_2$, and $B\_2$ is $n\_2$-dimensional. Each of the output $n\_2$-dimensional vectors is conceptually a representation of a high-resolution patch that will be used for reconstruction.

### Reconstruction
In the traditional methods, the predicted overlapping high-resolution patches are often averaged to produce the final full image. The averaging can be considered as a pre-defined filter on a set of feature maps (where each position is the "flattened" vector form of a high-resolution patch). Motivated by this, we define a convolutional layer to produce the final high-resolution image:
$$F(Y)=W\_3 * F\_2(Y) + B\_3$$
Here W_3 corresponds to $c$ filters of a size $n\_2 \times f\_3 \times f\_3$, and $B\_3$ is a $c$-dimensional vector.

## Training
Loss function: given a set of high-resolution images ${X\_i}$ and their corresponding low-resolution images ${Y\_i}$, we use mean squared error (MSE) as the loss function:
$$L(\Theta ) = \frac{1}{n}\sum^n\_{i=1}\Vert F(Y\_i;\Theta)-X\_i\Vert ^2$$
where $n$ is the number of training samples. Using MSE as the loss function favors a high PSNR. The PSNR is widely-used metric for quantitatively evaluating image restoration quality, and is at least partially related to the perceptual quality. Despite that the proposed model is trained favoring a high PSNR, we still observe satisfactory performance when the model is evaluated using alternative evaluation metrics, e.g., SSIM, MSSIM.
PSNR: Peak Signal to Noise Ratio. 是一种评价图像的客观标准。
$$PSNR = 10 \times \log\_{10}(\frac{(2^n-1)^2}{MSE})$$
其中，MSE是原图像和处理图像之间的均方误差，n是每个采样值的比特数，单位是dB。
The loss is minimized using stochastic gradient descent with the standard backpropagation. In particular, the weight matrices are updated as
$$\Delta\_{i+1}=0.9 \cdot \Delta\_i + \eta \cdot \frac{\partial{L}}{\partial{W^\ell\_i}}, W^\ell\_{i+1}=W^\ell\_{i}+\Delta\_{i+1}$$
where $\ell \in {1,2,3}$ and $i$ are the indices of layers and iterations, $\eta$ is the learning rate, and $\frac{\partial{L}}{\partial{W^\ell\_i}}$ is the derivative. The filter weights of each layer are initialized by drawing randomly from a Gaussian distribution with zero mean and standard deviation 0.0001 (and 0 for biases). The learning rate is $10^{-4}$ for the first two layers, and $10^{-5}$ for the last layer. We empirically find that a smaller learning rate in the last layer is important for the network to converge (similar to the denoising case).
In the training phase, the ground truth images ${X\_i}$ are prepared as $f\_{sub} \times f\_{sub} \times c$-pixel sub-images randomly cropped from the training images. By "sub-images" we mean these samples are treated as small "images" rather than "patches", in the sense that "patches" are overlapping and require some averaging as post-processing but "sub-images" need not. To synthesize the low-resolution samples ${Y\_i}$, we blur a sub-image by a Gaussian kernel, sub-sample it by the upscaling factor, and upscale it by the same factor via bicubic interpolation.
To avoid border effects during training, all the convolutional layers have no padding, and the network produces a smaller output $((f\_{sub}-f\_1-f\_2-f\_3+3)^2 \times c)$. The MSE loss function is evaluated only by the difference between the contral pixels of $X\_i$ and the network output. Although we use a fixed image size in training, the convolutional nerual network can be applied on images of arbitrary sizes during testing.