## Overview
Softmax regression is a type of logistic regression that extends it to a multiclass problem by outputting a vector $P$ of probabilities for each distinct class and taking $argmax(P)$.

## Connection to a regular logistic regression
Recall that a standard logistic regression is aimed at approximating
$$
p = \frac{1}{e^{-X\beta}+1} = \\
= \frac{e^{X\beta}}{1+e^{X\beta}},
$$

which actually alignes with the definition of the softmax function:
$$
softmax(z_i)=\sigma(z_i)=\frac{e^{z_i}}{\sum_j^Ce^{z_j}},
$$

where $C$ is the number of classes and values of which sum up to $1$. Hence it simply extends the functionality of sigmoid to more than 2 classes and could be used for assigning probability values in a categorical distribution, i.e. softmax regression searches for the following vector-approximation:
$$
p^{(i)}=\frac{e^{x^{(i)}\beta}}{\sum_j^Ce^{x^{(i)}\beta_j}_j}
$$

## Loss in softmax regression
**tl;dr** key differences in the loss from logistic regression include replacing sigmoid with softmax and calculating several gradients for vectors $\beta_j$ corresponding to a particular class $j\in\{1,...,C\}$.

Recall that we use MLE in logistic regression. It is the same case with softmax regression, although instead of Bernoulli-distributed random variable we have categorical distribution, which is an extension of Bernoulli to more than 2 labels. Its PMF is defined as:
$$
f(y|p)=\prod_{i=1}^Kp_i^{[i=y]},
$$

Hence, our log-likelihood looks like:
$$
\sum_X \sum_j^C [y_i=j] \log \left[p\left(x_i\right)\right]
$$

Where we replace our probability function with softmax:
$$
\sum_X \sum_j^C [y_i=j] \log \frac{e^{x_i\beta_j}}{\sum_j^Ce^{x_i\beta_j}}
$$

where $[i=y]$ is a function, that returns $0$, if $i\neq y$, and $1$ otherwise and $C$ - number of distinct classes (labels). You can see that since we are expecting a $1\times C$ output of $y$, just like in the neuron backprop problem, we will be having separate vector $\beta_j$ for every $j$ class out of $C$. 

## Optimization objective
The optimization objective is the same as with logistic regression. The function, which we are optimizing, is also commonly refered as **Cross Entropy** (CE):

$$
argmin_\beta -[\sum_X \sum_j^C [y_i=j] \log \frac{e^{x_i\beta_j}}{\sum_j^Ce^{x_i\beta_j}}] \\
$$

Then we are yet again using a chain rule for calculating partial derivative of $CE$ with respect to $\beta$:


$$
\frac{\partial CE}{\partial\beta^{(j)}_i}=\frac{\partial CE}{\partial\sigma}\frac{\partial\sigma}{\partial[X\beta^{(j)}]}\frac{\partial[X\beta^{(j)}]}{\beta^{(j)}_i}
$$

Which is eventually reduced to a similar to logistic regression gradient matrix form:
$$
X^T(\sigma(X\beta^{(j)})-Y)
$$

Then we can finally use gradient descent in order to iteratively update our parameters with respect to a particular class:
$$
\beta^{(j)}_{t+1}=\beta^{(j)}_t - \eta [X^T(\sigma(X\beta^{(j)}_t)-Y)]
$$
