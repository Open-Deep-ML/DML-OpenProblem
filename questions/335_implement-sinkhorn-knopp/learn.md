
## Understanding the Sinkhorn–Knopp Algorithm for Manifold-Constrained Hyper-Connections

### Introduction

Residual connections stabilize deep neural networks by preserving identity mappings across layers. Hyper-connections extend this idea by allowing multiple residual streams to be mixed using a learnable matrix

- Residual update:
  $$
  h_{l+1} = h_l + f(h_l)
  $$
- Hyper-connections update:
  $$
  h_{l+1} = H_{\text{res}} . h_l + f(h_l)
  $$

While hyper-connections increase expressivity, an unconstrained mixing matrix $ H_{\text{res}} $ can cause signal explosion or vanishing when applied repeatedly across many layers.

### Manifold-Constrained Hyper-Connections (mHC)

To ensure stable signal propagation, manifold-constrained hyper-connections restrict the residual mixing matrix to lie in the Birkhoff polytope, the set of doubly stochastic matrices

- Row sums equal to 1:
  $$ 
  \sum_j H_{ij} = 1 
  $$
- Column sums equal to 1
  $$ 
  \sum_i H_{ij} = 1 
  $$
- Non-negativity:
  $$ 
  H_{ij} \ge 0 
  $$

These constraints ensure that residual streams are redistributed rather than amplified, preserving signal norms while enabling rich feature mixing.

### Sinkhorn–Knopp Algorithm

The Sinkhorn–Knopp algorithm transforms a matrix into a doubly stochastic matrix by alternately normalizing its rows and columns. Since the input matrix may contain negative values, it is first mapped to a non-negative matrix using an element-wise exponential
 $$ 
 H^{(0)} = \exp(H_{\text{res}}) 
 $$

The algorithm then performs the following steps iteratively:

1. **Row normalization**
  $$ 
  H^{'}{ij} = \dfrac{H^{(k)}{ij}}{\sum_j H^{(k)}_{ij}} 
  $$
2. **Column normalization**
  $$ 
  H^{(k+1)}{ij} = \dfrac{H^{'}{ij}}{\sum_i H^{'}_{ij}} 
  $$

After sufficient iterations, the matrix converges to a doubly stochastic matrix.

### Why Doubly Stochastic?

 - Row sums = 1 → prevents signal amplification
 - Column sums = 1 → prevents information loss

Together, these properties restore the norm-preserving behavior of residual connections, while retaining the expressive power of hyper-connections.

### Key Takeaways
 - Hyper-connections improve expressivity but can destabilize deep networks.
 - Manifold constraints enforce stable, balanced residual mixing.
 - The Sinkhorn–Knopp algorithm efficiently projects matrices onto the Birkhoff polytope. 
 - Exponentiation allows handling unconstrained (including negative) matrices.
    