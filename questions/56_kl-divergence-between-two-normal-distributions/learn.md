## Understanding Kullback-Leibler Divergence (KL Divergence)

The **Kullback-Leibler (KL) divergence**, also known as relative entropy, measures the difference between two probability distributions. It quantifies how much information is lost when approximating one distribution with another.

---

### Definition of KL Divergence

For continuous variables, the KL divergence is defined as:

$$
KL(P \parallel Q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx
$$

where:
- $p(x)$ is the probability density function of the **reference** distribution $P$.
- $q(x)$ is the probability density function of the **comparison** distribution $Q$.

---

### KL Divergence Between Two Normal Distributions

Consider two normal distributions $P$ and $Q$:

- $P \sim N(\mu_P, \sigma_P^2)$  
- $Q \sim N(\mu_Q, \sigma_Q^2)$

For these, the KL divergence simplifies to:

$$
KL(P \parallel Q) = \int p(x) \left[
\log \frac{\sigma_Q}{\sigma_P}
+ \frac{\sigma_P^2 + (\mu_P - \mu_Q)^2}{2\sigma_Q^2}
- \frac{1}{2}
\right] dx
$$

Since $p(x)$ is the PDF of $x$ under $P$, the integral over $p(x)$ just multiplies by 1 for each constant term. Thus, the final closed form is:

$$
KL(P \parallel Q) =
\log \frac{\sigma_Q}{\sigma_P}
+ \frac{\sigma_P^2 + (\mu_P - \mu_Q)^2}{2\sigma_Q^2}
- \frac{1}{2}
$$

---

### Interpretation

This expression quantifies how one normal distribution $P$ **diverges** from another normal distribution $Q$. A KL divergence of zero indicates the two distributions are identical. As the divergence grows, it signals that $Q$ is a poorer approximation of $P$.

The KL divergence is **asymmetric**:

$$
KL(P \parallel Q) \neq KL(Q \parallel P)
$$

making it sensitive to the **direction** of comparison.
