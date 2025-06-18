## Understanding Global Average Pooling

Global Average Pooling (GAP) is a pooling operation commonly used in convolutional neural networks (CNNs) to reduce the spatial dimensions of feature maps. Unlike traditional pooling methods like max pooling or average pooling with a fixed window size, GAP computes the average of each entire feature map, resulting in a single value per channel.

### How It Works

Given a 3D input tensor of shape $(H, W, C)$, where:
- $H$ is the height,
- $W$ is the width,
- $C$ is the number of channels (feature maps),

Global Average Pooling produces a 1D output vector of shape $(C,)$, where each element is the average of all values in the corresponding feature map.

Mathematically, for each channel $c$:

$$
\text{GAP}(x)_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{i,j,c}
$$

### Benefits of Global Average Pooling

- **Parameter Reduction**: By replacing fully connected layers with GAP, the number of parameters is significantly reduced, which helps in preventing overfitting.
- **Spatial Invariance**: GAP captures the global information from each feature map, making the model more robust to spatial translations.
- **Simplicity**: It is a straightforward operation that doesn't require tuning hyperparameters like pooling window size or stride.

### Use in Modern Architectures

Global Average Pooling is a key component in architectures like ResNet, where it is used before the final classification layer. It allows the network to handle inputs of varying sizes, as the output depends only on the number of channels, not the spatial dimensions.

### Example

Consider a 2x2x3 input tensor:

$$
x = \begin{bmatrix}
\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix},
\begin{bmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{bmatrix}
\end{bmatrix}
$$

Applying GAP:

- For channel 0: $\frac{1+4+7+10}{4} = \frac{22}{4} = 5.5$
- For channel 1: $\frac{2+5+8+11}{4} = \frac{26}{4} = 6.5$
- For channel 2: $\frac{3+6+9+12}{4} = \frac{30}{4} = 7.5$

Thus, the output is $[5.5, 6.5, 7.5]$.

This operation effectively summarizes each feature map into a single value, capturing the essence of the features learned by the network.
