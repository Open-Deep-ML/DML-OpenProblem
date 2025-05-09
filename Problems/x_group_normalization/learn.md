## Understanding Group Normalization

Group Normalization (GN) is a normalization technique that divides the channels into groups and normalizes the activations within each group. Unlike Batch Normalization, which normalizes over the entire mini-batch, Group Normalization normalizes over groups of channels and is less dependent on the batch size. This makes it particularly useful for tasks with small batch sizes or when using architectures such as segmentation networks where spatial resolution is important.

### Concepts

Group Normalization operates on the principle of normalizing within smaller groups of channels. The process reduces **internal covariate shift** within these groups and helps stabilize training, especially in scenarios where the batch size is small or varies across tasks.

The process of Group Normalization consists of the following steps:

1. **Divide Channels into Groups:** Split the feature channels into several groups. The number of groups is determined by the **n_groups** parameter.
2. **Compute the Mean and Variance within Each Group:** For each group, compute the mean and variance of the activations within the group, across the spatial dimensions and batch.
3. **Normalize the Inputs:** Normalize the activations of each group using the computed mean and variance.
4. **Apply Scale and Shift:** After normalization, apply a learned scale (gamma) and shift (beta) to restore the model's ability to represent the data's original distribution.

### Structure of Group Normalization for BCHW Input

For an input tensor with the shape **BCHW** , where:
- **B**: batch size,
- **C**: number of channels,
- **H**: height,
- **W**: width,
the Group Normalization process operates on specific dimensions based on the task's requirement.

#### 1. Group Division

- The input feature dimension **C** (channels) is divided into several groups. The number of groups is determined by the **n_groups** parameter, and the size of each group is calculated as:
  
  $$
  \text{groupSize} = \frac{C}{n_{\text{groups}}}
  $$

  Where:
  - **C** is the number of channels.
  - **n_groups** is the number of groups into which the channels are divided.
  - **groupSize** is the number of channels in each group.

  The input tensor is then reshaped to group the channels into the specified groups.

#### 2. Mean and Variance Calculation within Groups

- For each group, the **mean** $\mu_g$ and **variance** $\sigma_g^2$ are computed over the spatial dimensions and across the batch. This normalization helps to stabilize the activations within each group.
  
  $$ 
  \mu_g = \frac{1}{B \cdot H \cdot W \cdot \text{groupSize}} \sum_{i=1}^{B} \sum_{h=1}^{H} \sum_{w=1}^{W} \sum_{g=1}^{\text{groupSize}} x_{i,g,h,w}
  $$

  $$
  \sigma_g^2 = \frac{1}{B \cdot H \cdot W \cdot \text{groupSize}} \sum_{i=1}^{B} \sum_{h=1}^{H} \sum_{w=1}^{W} \sum_{g=1}^{\text{groupSize}} (x_{i,g,h,w} - \mu_g)^2
  $$

  Where:
  - $x_{i,g,h,w}$ is the activation at batch index $i$, group index $g$, height $h$, and width $w$.
  - $B$ is the batch size.
  - $H$ and $W$ are the spatial dimensions (height and width).
  - $\text{groupSize}$ is the number of channels in each group.

#### 3. Normalization

Once the mean $\mu_g$ and variance $\sigma_g^2$ have been computed for each group, the next step is to **normalize** the input. The normalization is done by subtracting the mean and dividing by the standard deviation (square root of the variance, plus a small constant $\epsilon$ for numerical stability):

$$
\hat{x}_{i,g,h,w} = \frac{x_{i,g,h,w} - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}}
$$

Where:
- $\hat{x}_{i,g,h,w}$ is the normalized activation for the input at batch index $i$, group index $g$, height $h$, and width $w$.
- $\epsilon$ is a small constant to avoid division by zero.

#### 4. Scale and Shift

After normalization, the next step is to apply a **scale** ($\gamma_g$) and **shift** ($\beta_g$) to the normalized activations for each group. These learned parameters allow the model to adjust the output distribution of each group:

$$
y_{i,g,h,w} = \gamma_g \hat{x}_{i,g,h,w} + \beta_g
$$

Where:
- $\gamma_g$ is the scaling factor for group $g$.
- $\beta_g$ is the shifting factor for group $g$.

#### 5. Training and Inference

- **During Training**: The mean and variance are computed for each mini-batch and used for normalization within each group.
- **During Inference**: The model uses running averages of the statistics (mean and variance) that were computed during training to ensure consistent behavior when deployed.

### Key Points

- **Group-wise Normalization**: Group Normalization normalizes within smaller groups of channels instead of normalizing over the entire batch and all channels. This allows for more stable training in cases with small batch sizes.
  
- **Number of Groups**: The number of groups is a hyperparameter (**n_groups**) that can significantly affect the model’s performance. It is typically set to divide the total number of channels into groups of equal size.
  
- **Smaller Batch Sizes**: Group Normalization is less dependent on the batch size, making it ideal for situations where batch sizes are small (e.g., segmentation tasks).

- **Numerical Stability**: As with other normalization techniques, a small constant $\epsilon$ is added to the variance to avoid numerical instability when dividing by the square root of variance.

- **Improved Convergence**: Group Normalization can help improve the gradient flow throughout the network, making it easier to train deep networks with small batch sizes. It also helps speed up convergence and stabilize training.

- **Regularization Effect**: Similar to Batch Normalization, Group Normalization introduces a form of regularization through the normalization process. It can reduce overfitting by acting as a noise source during training.

### Why Normalize Over Groups?

- **Group-wise Normalization**: By dividing the channels into smaller groups, Group Normalization ensures that each group has a stable distribution of activations, making it effective even when batch sizes are small.
  
- **Less Dependency on Batch Size**: Unlike Batch Normalization, Group Normalization does not require large batch sizes to compute accurate statistics. This makes it well-suited for tasks such as image segmentation, where large batch sizes may not be feasible.

- **Channel-wise Learning**: Group Normalization allows each group to learn independently, preserving flexibility while also controlling the complexity of normalization over channels.

By normalizing over smaller groups, Group Normalization can reduce internal covariate shift and allow for faster and more stable training, even in situations where Batch Normalization may be less effective due to small batch sizes.