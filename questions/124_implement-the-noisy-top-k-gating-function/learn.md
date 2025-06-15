## Noisy Top-K Gating

Noisy Top-K Gating is a sparse selection mechanism used in Mixture-of-Experts (MoE) models. It routes input tokens to a subset of available experts, enhancing efficiency and model capacity.

### Overview

The core idea is to add learned noise to the gating logits and then select only the top-k experts for each input. This encourages exploration and helps balance load across experts.

### Step-by-Step Breakdown

1. **Compute Raw Gate Scores**  
   First, compute two linear projections of the input:
   $$
   H_{\text{base}} = X W_g
   $$
   $$
   H_{\text{noise}} = X W_{\text{noise}}
   $$

2. **Apply Noise with Softplus Scaling**  
   Add pre-sampled Gaussian noise, scaled by a softplus transformation:
   $$
   H = H_{\text{base}} + N \odot \text{Softplus}(H_{\text{noise}})
   $$

3. **Top-K Masking**  
   Keep only the top-k elements in each row (i.e., per input), setting the rest to $-\infty$:
   $$
   H' = \text{TopK}(H, k)
   $$

4. **Softmax Over Top-K**  
   Normalize the top-k scores into a valid probability distribution:
   $$
   G = \text{Softmax}(H')
   $$

### Worked Example

Let:
- $X = [[1.0, 2.0]]$
- $W_g = [[1.0, 0.0], [0.0, 1.0]]$
- $W_{\text{noise}} = [[0.5, 0.5], [0.5, 0.5]]$
- $N = [[1.0, -1.0]]$
- $k = 2$

Step-by-step:
- $H_{\text{base}} = [1.0, 2.0]$
- $H_{\text{noise}} = [1.5, 1.5]$
- $\text{Softplus}(H_{\text{noise}}) \approx [1.804, 1.804]$
- $H = [1.0 + 1.804, 2.0 - 1.804] = [2.804, 0.196]$
- Softmax over these gives: $[0.917, 0.0825]$

### Benefits

- **Computational Efficiency**: Activates only k experts per input.
- **Load Balancing**: Injected noise encourages diversity in expert selection.
- **Improved Generalization**: Acts as a regularizer via noise-based gating.

This technique is used in large sparse models like GShard and Switch Transformers.
