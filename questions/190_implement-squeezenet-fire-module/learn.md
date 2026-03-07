## SqueezeNet Fire Module Explained

SqueezeNet achieves AlexNet-level accuracy with 50x fewer parameters through its "Fire Module".

### Architecture

The Fire Module consists of two layers:

1. **Squeeze Layer**: A convolution layer with only 1x1 filters.
   - Purpose: Reduce the number of input channels before feeding them to larger 3x3 filters. This minimizes computation and parameters.
   - Notation: $s_{1x1}$ is the number of 1x1 filters in the squeeze layer.

2. **Expand Layer**: A mix of 1x1 and 3x3 convolution filters.
   - Purpose: Capture both local details (3x3) and pointwise features (1x1).
   - Notation:
     - $e_{1x1}$: Number of 1x1 filters in expand layer.
     - $e_{3x3}$: Number of 3x3 filters in expand layer.
   - The outputs of these two parallel convolutions are concatenated along the channel dimension.

### Operation

The flow is:
`Input -> Squeeze (1x1 conv + ReLU) -> Expand ( (1x1 conv + ReLU) || (3x3 conv + ReLU) ) -> Concatenate -> Output`

### Why it works?

Standard 3x3 convolution on $C_{in}$ channels with $C_{out}$ filters has parameters:
$P_{standard} = 3 \times 3 \times C_{in} \times C_{out}$

In Fire Module:

1. Squeeze layer reduces $C_{in}$ to $s_{1x1}$ (small). Parameters: $1 \times 1 \times C_{in} \times s_{1x1}$.
2. Expand 3x3 layer operates on only $s_{1x1}$ channels. Parameters: $3 \times 3 \times s_{1x1} \times e_{3x3}$.
3. Expand 1x1 layer adds more features cheaply. Parameters: $1 \times 1 \times s_{1x1} \times e_{1x1}$.

Total parameters are significantly lower if $s_{1x1} \ll C_{in}$.

### Implementation Details

- **Padding**: Use 'same' padding for 3x3 convolutions so spatial dimensions match 1x1 outputs.
- **Concatenation**: Join along the channel axis (axis=-1 or 1 depending on format, here we assume (H, W, C)).
