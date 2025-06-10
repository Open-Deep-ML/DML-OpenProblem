## KL divergence and its properties
KL divergence is used as a measure of dissimilarity between two distributions. It is defined by the following formula:
$$
D_{KL}(P || Q) = \mathbb{E}_{x\sim P(X)}log\frac{P(X)}{Q(X)},
$$
where $P(X)$ observed distribution we compare everything else with and $Q(X)$ is usually the varying one; $P(X)$ and $Q(X)$ are PMF (but could also be denoted as PDFs $f(x)$ and $q(x)$ in continuos case). The function has following properties:
* $D_{KL}\geq0$
* assymetry: $D_{KL}(P || Q) \neq D_{KL}(Q || P)$

## Finding $D_{KL}$ between two multivariate Gaussians
Consider two multivariate Normal distributions:
$$
p(x)\sim \mathbb{N}(\mu_1,\Sigma_1), \\
q(x)\sim \mathbb{N}(\mu_2,\Sigma_2)
$$

PDF of a multivariate Normal distribution is defined as:
$$
f(x)=\frac{1}{(2\pi)^\frac{p}{2}|\Sigma|^\frac{1}{2}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)),
$$

where $\Sigma$ - covariance matrix, $|\cdot|$ - determinant, $p$ - size of the random vector, i.e. number of different normally distributed features inside $P$ and $Q$ and $x$ usually denotes $x^T$, which is a random vector of size $p\times1$.

Now we can move onto calculating KL divergence for these two distributions, skipping the division part of two PDFs:
$$
\frac{1}{2}[\mathbb{E_p}log\frac{|\Sigma_q|}{|\Sigma_p|} ^ \textbf{[1]} - \mathbb{E_p}(x-\mu_p)^T\Sigma_p^{-1}(x-\mu_p) ^ \textbf{[2]} + \\
+ \mathbb{E_p}(x-\mu_q)^T\Sigma_q^{-1}(x-\mu_q) ^ \textbf{[3]}]= \\
= \frac{1}{2}[log\frac{|\Sigma_q|}{|\Sigma_p|}-p+(\mu_p-\mu_q)^T\Sigma^{-1}_q(\mu_p-\mu_q) + \\
+ tr(\Sigma^{-1}_q\Sigma_p)],
$$
where in order to achieve an equality we proceed to do $\textbf{[1]}:$
$$
log\frac{|\Sigma_q|}{|\Sigma_p|}=const\implies \text{EV equals to the value itself;}
$$

then $\textbf{[2]}:$
$$
\underset{N \times p}{(x-\mu_p)^T} * \sum_{p \times p} * \underset{N \times p}{(x-\mu_p)^T} = \underset{N\times N}{A}\text {, where } N=1 \implies \\
\implies A=\operatorname{tr}(A)
$$

Recall that:
$$
\operatorname{tr}(A B C)=\operatorname{tr}(B C A)=\operatorname{tr}(C B A)
$$

Then:
$$
\operatorname{tr}(A)=\operatorname{tr}\left(\left(x-\mu_p\right)^{\top}\left(x-\mu_p\right) \Sigma_p^{-1}\right)\\ =\operatorname{tr}\left(\Sigma_p \Sigma_p^{-1}\right)=\operatorname{tr}(I)=p  
$$

and finally $\textbf{[3]}$, where we should recall, that for multivariate Normal distributions this is true ($x\sim\mathbb{N}(\mu_2, \Sigma_2)$):
$$
\mathbb{E}(x-\mu_1)^TA(x-\mu_1)= \\
= (\mu_2-\mu_1)^TA(\mu_2-\mu_1)+tr(A\Sigma_2)
$$