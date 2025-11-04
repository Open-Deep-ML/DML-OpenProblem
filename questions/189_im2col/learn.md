
# Learn Section

#  What is Im2col?
- Also known as Image to Column, *im2col* is a patch extraction operation designed to convert a convolution into a general matrix multiplication (GEMM) problem.  It reorganizes overlapping image patches into rows of a 2D matrix so that the convolution kernel can be applied using efficient linear algebra routines.

## Why are we adding a step to convert into matrix multiplication instead of directly doing element-wise convolution?
- Decades of optimization have gone into GPU and CPU matrix multiplication libraries (e.g., cuBLAS, MKL). By reformulating convolution as GEMM, we can leverage these highly optimized routines.
    - Element-wise on the other hand, performs worse due to the irregular memory accesses during the convolution operation.
- In the im2col layout, pixels are arranged contiguously in memory, enabling faster and more predictable access patterns.
- Although im2col introduces some data redundancy (overlapping patches share values), the performance gains from contiguous memory access and GEMM vastly outweigh this cost.

# Implementation of Im2col
A simple way to visualize im2col is to imagine **sliding a kernel window** across an image and recording each window as one row of a new matrix.

For example, a 3x3 kernel sliding over a 6x6 image with a stride of 1, would produce a matrix of shape `(9,16)`. This is because the convolution output has:

$$
H_{out} = [\frac{H_{in} - k_{h}}{s}] + 1
$$
$$
W_{out} = [\frac{W_{in} - k_{w}}{s}] + 1
$$

and the flattened patch dimension is:
$$
I_{out} = (C_{in} \cdot k_h \cdot k_w, H_{out} \cdot W_{out})
$$

Where:
- $H_{in}, W_{in}$: Input image dimensions
- $k_h, k_w$: Kernel (filter) height and width
- $s$: Stride
- $C_{in}$: Number of input channels

> **Note:** Padding is assumed to be 0 in these equations

By flattening each local patch, the convolution becomes equivalent to a matrix multiplication between the flattened image matrix (im2col output) and the flattened kernel weights, which allows for GEMM optimizations.
    