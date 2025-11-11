
# Learn Section

## Understanding Pooling

In computer vision and image processing, padding refers to the process of adding pixels (or matrix values) around the border of in input to preserve its dimension and prevent information loss during processing by, for example, and convolutional neural network (CNN). There are different strategies for padding, such as the position where padding is added (ie. left of image vs. around the image), the amount of padding added to each different axis, and even the specific value that is used as the padding value. In this simple example, the padding is done around the entire image, and each axis gets the same amount of padding.

Consider the following input matrix:

$$
\begin{bmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{bmatrix}
$$

If we consider a padding of 1, the padded matrix would have an additional "1 pixel" of values (zeros in this case) around it. This would look like: 

$$
\begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & a_{11} & a_{12} & 0 \\
0 & a_{21} & a_{22} & 0 \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

With a padding of 2, there would be "2 pixels" around the picture, and so on. If the input matrix happened to be a rectangle the same logic will apply. Though in practice there may be more ways to customize this process, this simple idea of padding can help maintain certain dimensions of data throughout image processing and help reduce information loss at the edges of an image. 

    