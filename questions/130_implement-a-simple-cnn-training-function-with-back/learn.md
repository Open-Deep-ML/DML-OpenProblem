## Understanding a Simple Convolutional Neural Network with Backpropagation

A **Convolutional Neural Network** (CNN) learns two things at once:  
1. **What to look for** - small filters (kernels) that detect edges, textures, etc.  
2. **How to combine those detections** - a dense layer that converts them into class probabilities.

Below is the full training loop broken into intuitive steps that can be implemented directly in NumPy.

---

### 1. Forward Pass

**Convolution**  
The convolution layer slides a small filter over the input and produces feature maps:

$$
Z^c[p, q, k] = \sum_{i, j} X[p+i, q+j] \cdot W^c[i, j, k] + b^c[k]
$$

This results in a tensor of shape $(H - k + 1, W - k + 1, F)$, where $H$ and $W$ are the input height and width, $k$ is the kernel size, and $F$ is the number of filters.

**ReLU Activation**

$$
A^c = \max(0, Z^c)
$$

This introduces non-linearity by zeroing out negative values.

**Flattening**

The feature maps are reshaped into a vector:

$$
A^f = \text{flatten}(A^c)
$$

**Dense Layer**

$$
Z^d = A^f \cdot W^d + b^d
$$

Each entry in $A^f$ contributes to every output class via weight matrix $W^d$ and bias $b^d$.

**Softmax Activation**

$$
\hat{y}_c = \frac{e^{Z^d_c}}{\sum_j e^{Z^d_j}}
$$

This converts raw scores into probabilities for classification.

---

### 2. Loss Function – Cross Entropy

For one-hot encoded label $y$ and prediction $\hat{y}$:

$$
\mathcal{L}(\hat{y}, y) = -\sum_c y_c \log(\hat{y}_c)
$$

This penalizes incorrect predictions based on confidence.

---

### 3. Backward Pass

**Gradient of Softmax + Cross Entropy**

$$
\frac{\partial \mathcal{L}}{\partial Z^d} = \hat{y} - y
$$

**Dense Layer Gradients**

$\frac{\partial \mathcal{L}}{\partial W^d} = (A^f)^T \cdot \frac{\partial \mathcal{L}}{\partial Z^d}$,
and the gradient with respect to biases is
$\frac{\partial \mathcal{L}}{\partial b^d} = \frac{\partial \mathcal{L}}{\partial Z^d}$.

Reshape the upstream gradient to the shape of $A^c$ for backpropagation through ReLU.

**ReLU Gradient**

$$
\frac{\partial \mathcal{L}}{\partial Z^c} = \frac{\partial \mathcal{L}}{\partial A^c} \cdot \mathbf{1}(Z^c > 0)
$$

**Convolution Filter Gradients**

For each filter $k$:

$$
\frac{\partial \mathcal{L}}{\partial W^c_{i,j,k}} = \sum_{p,q} \frac{\partial \mathcal{L}}{\partial Z^c_{p,q,k}} \cdot X_{p+i, q+j}
$$

$$
\frac{\partial \mathcal{L}}{\partial b^c_k} = \sum_{p,q} \frac{\partial \mathcal{L}}{\partial Z^c_{p,q,k}}
$$

---

### 4. Updating Parameters (SGD)

With learning rate $\eta$:

$$
W \leftarrow W - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}
$$

$$
b \leftarrow b - \eta \cdot \frac{\partial \mathcal{L}}{\partial b}
$$

Repeat this process for each sample (stochastic gradient descent) and for multiple epochs.

---

### 5. Example Walkthrough

Suppose $X$ is a grayscale image:

$$
X = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
$$

And the kernel is:

$$
K = \begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
$$

Perform convolution at the top-left:

$$
(1 \cdot 1 + 2 \cdot 0 + 4 \cdot 0 + 5 \cdot (-1)) = 1 - 5 = -4
$$

After ReLU: max(0, -4) = 0  
Flatten the result -> Dense layer -> Softmax output -> Compute loss

Backpropagate the error to adjust weights, and repeat to learn better filters and classifications over time.
