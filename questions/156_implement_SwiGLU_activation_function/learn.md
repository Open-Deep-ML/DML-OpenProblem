## Understanding the SwiGLU Activation Function

As the name suggests the SwiGLU activation function is a combination of two activations - Swish (implemented as SiLU in PyTorch) and GLU (Gated Linear Unit). It is important that we understand Swish and GLU because SwiGLU inherits properties from both — the smooth self-gating behavior of Swish, the decoupled gating structure of GLU.

### Swish Activation (Self-Gating)

**Swish**, introduced by Google Brain, is a smooth, self-gated activation function defined as:

$$
\text{Swish}(x) = x \cdot \sigma(x)
$$

where the sigmoid function is:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

In Swish, the same input $x$ is used to:
  - **Compute the gate**: $\sigma(x)$
  - **Modulate itself**: $x \cdot \sigma(x)$

This is called **self-gating** — the input both **creates** and **passes through** the gate. \
**Note:** When written in a PyTorch forward loop, it looks something like -
```bash
import torch.nn.functional as F

def forward(self, x):
   x1 = self.fc1(x)   # x1 = Wx + b where W, b are learnable params
   output = F.silu(x) # output = x1 * sigmoid(x1) 
   return output      # output = (Wx + b) * sigmoid(Wx + b)
```
This essentially means that the gate is learnable, and the model learns the best shape of the activation function.

### Gated Linear Unit (GLU)

**GLU**, introduced in *Language Modeling with Gated Convolutional Networks* (Dauphin et al., 2017), is a gated activation mechanism defined as:

$$
\text{GLU}(x_1, x_2) = x_1 \cdot \sigma(x_2)
$$

Here:
- $x_1$ is the **input signal**.
- $x_2$ is used to **compute the gate** via the sigmoid function.

In practice, both $x_1$ and $x_2$ are obtained by **splitting the output of a single linear layer**:

```bash
import torch.nn.functional as F

def forward(self, x):
   x_proj = self.fc1(x)                
   x1, x2 = x_proj.chunk(2, dim=-1)    # x1 = Wx + b, x2 = Vx + c
   output = x1 * torch.sigmoid(x2)     # GLU = x1 · σ(x2)
   return output
```
So GLU can be rewritten as:

$$
\text{GLU}(x) = x_1 \cdot \sigma(x_2)
$$
where:
$$x_1 = W x + b$$
 $$x_2 = V x + c$$


This is a learned, cross-gating mechanism — the model learns different parameters for the signal and the gate.


## SwiGLU

With Swish and GLU out of the way, it becomes very easy to understand **SwiGLU**. It is defined as:

$$
\text{SwiGLU}(x) = x_1 \cdot \text{Swish}(x_2)
$$

Where:
- $x_1, x_2$ are typically obtained by splitting a linear projection of the input (inspired by GLU).

- $\text{Swish}(x_2) = x_2 \cdot \sigma(x_2)$ is the self-gated activation.

So putting it together:

$$
\text{SwiGLU}(x) = x_1 \cdot (x_2 \cdot \sigma(x_2))
$$

This combines the **signal-gate decoupling** of GLU with the **smooth self-gating** of Swish, and is used in the feed-forward blocks of large-scale models like Google's PaLM, Meta's LLaMA.


### Why Does It Work?
> Noam Shazeer, the author in his paper writes: "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."

The improvement in performance have only been proven *emprically* by observing faster convergence during training