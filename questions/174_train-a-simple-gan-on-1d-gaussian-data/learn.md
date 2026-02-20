## Understanding GANs for 1D Gaussian Data
A Generative Adversarial Network (GAN) consists of two neural networks - a **Generator** $G_\theta$ and a **Discriminator** $D_\phi$ - trained in a minimax game.

### 1. The Roles
- **Generator** $G_\theta(z)$: Takes a latent noise vector $z \sim \mathcal{N}(0, I)$ and outputs a sample intended to resemble the real data.
- **Discriminator** $D_\phi(x)$: Outputs a probability $p \in (0, 1)$ that the input $x$ came from the real data distribution rather than the generator.

### 2. The Objective
The classical GAN objective is:
$$
\min_{\theta} \; \max_{\phi} \; \mathbb{E}_{x \sim p_{\text{data}}} [\log D_\phi(x)] + \mathbb{E}_{z \sim p(z)} [\log (1 - D_\phi(G_\theta(z)))]
$$
Here:
- $p_{\text{data}}$ is the real data distribution.
- $p(z)$ is the prior distribution for the latent noise (often standard normal).

### 3. Practical Losses
In implementation, we minimize:
- **Discriminator loss**:
$$
\mathcal{L}_D = - \left( \frac{1}{m} \sum_{i=1}^m \log D(x^{(i)}_{\text{real}}) + \log(1 - D(x^{(i)}_{\text{fake}})) \right)
$$
- **Generator loss** (non-saturating form):
$$
\mathcal{L}_G = - \frac{1}{m} \sum_{i=1}^m \log D(G(z^{(i)}))
$$

### 4. Forward/Backward Flow
1. **Discriminator step**: Real samples $x_{\text{real}}$ and fake samples $x_{\text{fake}} = G(z)$ are passed through $D$, and $\mathcal{L}_D$ is minimized w.r.t. $\phi$.
2. **Generator step**: Fresh $z$ is sampled, $x_{\text{fake}} = G(z)$ is passed through $D$, and $\mathcal{L}_G$ is minimized w.r.t. $\theta$ while keeping $\phi$ fixed.

### 5. Architecture for This Task
- **Generator**: Fully connected layer ($\mathbb{R}^{\text{latent\_dim}} \to \mathbb{R}^{\text{hidden\_dim}}$) -> ReLU -> Fully connected layer ($\mathbb{R}^{\text{hidden\_dim}} \to \mathbb{R}^1$).
- **Discriminator**: Fully connected layer ($\mathbb{R}^1 \to \mathbb{R}^{\text{hidden\_dim}}$) → ReLU → Fully connected layer ($\mathbb{R}^{\text{hidden\_dim}} \to \mathbb{R}^1$) → Sigmoid.

### 6. Numerical Tips
- Initialize weights with a small Gaussian ($\mathcal{N}(0, 0.01)$).
- Add $10^{-8}$ to logs for numerical stability.
- Use a consistent batch size $m$ for both real and fake samples.
- Always sample fresh noise for the generator on each update.

**Your Task**: Implement the training loop to learn the parameters $\theta$ and $\phi$, and return the trained `gen_forward(z)` function. The evaluation (mean/std of generated samples) will be handled in the test cases.
