## Understanding Pixel Normalization
Pixel Normalization (PN) is a normalization technique that normalizes feature vectors at each spatial location across channels. Pixel Normalization is particularly useful in generative models such as Progressive GANs, where it helps control feature magnitudes and promotes consistent feature scaling during training.
### Mathematical Definition
For an input tensor with the shape **(B, C, H, W)**, where:
* B: batch size
* C: number of channels
* H: height
* W: width
The normalization for each pixel at spatial position *(h, w)* is computed as follows:
$$
x'_{b, c, h, w} = \frac{x_{b,c,h,w}}{\sqrt{\frac{1}{C}\sum_{i=1}^C x^2_{b,i,h,w}+\epsilon}}
$$
where:
* $x_{b,c,h,w}$ is the pixel value of channel *c* at position *(h, w)* for sample *b*.
* $\epsilon$ is a small constant added for numerical stability (e.g., $10^-8$).

This operation ensures that for every spatial *(h, w)*, the vector $[x'_{b, 1, h, w}, x'_{b, 2, h, w}, \ldots, x'_{b, C, h, w}]$ has unit norm, i.e:
$$
\frac{1}{C}\sum_{i=1}^C (x'_{b, i, h, w})^2 = 1
$$
### Why Pixel Normalization
* **Batch size independence**: Pixel Normalization does not rely on batch-level statistics such as mean or variance, making it suitable for training with very small batch sizes, even batch size = 1.
* **Training stability**: Removing batch dependencies leads to smoother convergence and more deterministic training behavior, especially in GANs.
* **Stable feature scaling**: By normalizing each pixel accross channels, it prevents the uncontrolled growth of activations, ensuring consistent feature magnitudes.
* **No parameters**: No learnable paramters ($\gamma$, $\beta$), reducing computational overhead while maintain effectiveness in deep generative networks.