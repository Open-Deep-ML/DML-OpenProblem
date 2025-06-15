A new study (https://arxiv.org/pdf/2503.10622) demonstrates that layer normalization, that is ubiquitous in transformers, produces Tanh-like S-shapes. By incorporating a new layer replacement for normalization called "Dynamic Tanh" (DyT for short), Transformers without normalization can match or exceed the performance of their normalized counterparts, mostly without hyperparameter tuning.

### Normalization layer
Consider an standard NLP task, where an input $x$ has a shape of $(B,T,C)$, where $B$ is the batch size, $T$ - number of tokens (sequence length) and $C$ - embedding dimensions. Then an output of a normalization layer is generally computed as $norm(x)=\gamma(\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}})+\beta$, where $\gamma$ and $\beta$ are learnable parameters of shape $(C,)$. Distribution's statistics are calculated as follows: $\mu_k=\frac{1}{BT}\sum_i^B\sum_j^Tx_{ij}$; $\sigma_k^2=\frac{1}{B T} \sum_{i, j}\left(x_{i j k}-\mu_k\right)^2$

### Hyperboloic tangent (Tanh)
Tanh function is defined as a ratio: $tanh(x)=\frac{sinh(x)}{cosh(x)}=\frac{exp(x)-exp(-x)}{exp(x)+exp(-x)}$. Essentially the function allows transformation of an arbitrary domain to $[-1,1]$. 

### Dynamic Tanh (DyT)
Turns out that LN (layer normalization) produces different parts of a $tanh(kx)$, where $k$ controls the curvature of the tanh curve in the center. The smaller the $k$, the smoother is the change from $-1$ to $1$. Hence the study proposes a drop-in replacement for LN given an input tensor $x$:

$$
DyT(x)=\gamma*tanh(\alpha x)+\beta,
$$

where:
* $\alpha$ - learnable parameter that allows scaling the input differently based on its range (tokens producing **smaller variance** produce **less smoother curves**). Authors suggest a **default value** of $0.5$.
* $\gamma, \beta$ - learnable parameters, that scale our output based on the input. Authors suggest initializing these vectors with following **default values**:
    * $\gamma$ as all-one vector 
    * $\beta$ as all-zero

Despite not calculating statistics, DyT preserves the "squashing" effect of LN on extreme values in a non-linear fashion, while almost linearly transforming central parts of the input.
