# **Gradient Checkpointing**

## **1. Definition**
Gradient checkpointing is a technique used in deep learning to reduce memory usage during training by selectively storing only a subset of intermediate activations (checkpoints) and recomputing the others as needed during the backward pass. This allows training of larger models or using larger batch sizes without exceeding memory limits.

## **2. Why Use Gradient Checkpointing?**
* **Reduce Memory Usage:** By storing fewer activations, memory requirements are reduced, enabling training of deeper or larger models.
* **Enable Larger Batches/Models:** Makes it possible to fit larger models or use larger batch sizes on limited hardware.
* **Tradeoff:** The main tradeoff is increased computation time, as some activations must be recomputed during the backward pass.

## **3. Gradient Checkpointing Mechanism**
Suppose a model consists of $N$ layers, each represented by a function $f_i$. Normally, the forward pass stores all intermediate activations:

$$
A_0 = x \\
A_1 = f_1(A_0) \\
A_2 = f_2(A_1) \\
\ldots \\
A_N = f_N(A_{N-1})
$$

With gradient checkpointing, only a subset of $A_i$ are stored (the checkpoints). The others are recomputed as needed during backpropagation. In the simplest case, you can store only the input and output, and recompute all intermediates when needed.

**Example:**
If you have three functions $f_1, f_2, f_3$ and input $x$:
* Forward: $A_1 = f_1(x)$, $A_2 = f_2(A_1)$, $A_3 = f_3(A_2)$
* With checkpointing, you might only store $x$ and $A_3$, and recompute $A_1$ and $A_2$ as needed.

## **4. Applications of Gradient Checkpointing**
Gradient checkpointing is widely used in training:
* **Very Deep Neural Networks:** Transformers, ResNets, and other architectures with many layers.
* **Large-Scale Models:** Language models, vision models, and more.
* **Memory-Constrained Environments:** When hardware cannot fit all activations in memory.
* **Any optimization problem** where memory is a bottleneck during training.

Gradient checkpointing is a powerful tool to enable training of large models on limited hardware, at the cost of extra computation.
