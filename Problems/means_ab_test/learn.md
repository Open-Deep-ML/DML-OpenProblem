## Overview:
A/B testing is a user experience research method. For example, we need to understand whether to deploy our model to production or not. For that reason we split our users into two groups, where the first one remains unchanged and the second one is being tested with the model's deployment. We calculate some metric to understand the statistical significance of the difference in it between two groups. If there is one, that means that we may gain an actual uplift by deploying our model. 

We calculate a two-sample statistic based on two distributions: $A$ (treatment group) and $B$ (controlled group), which we derive from a null hypothesis $H_0$, where we express our vision of how two distributions might be alike, and an alternative $H_1$, where we express otherwise. We then embed our hypothesis into some statistic, which follows some distribution $f(X)$. We then use this statistic and knowledge about a family of distributions for calculating probability $P(X\geq |t_{stat}| | H_0)=2\times(1-f(t))$ (for two-tailed distrbutions). This probability is called a P-value and it calculates the area under the PDF of $f(X)$ starting from $t_{stat}$, i.e. $\int_t^{max(X)}f(x)$. If the area falls under some threshold $\alpha$ (significance level), then we conclude that we do not have any reason for rejecting our null hypothesis. 

## When to use: Z-test vs T-test
One of the most widespread A/B tests for numeric i.i.d. (and for a better robustness normally distributed) variables (profit, conversion, e.t.c.) is used to compare two distributions of sample means under a null hypothesis of $H_0: \, \mu_A=\mu_B$. Therefore $H_1: \, \mu_A\neq\mu_B$. Depending on certain conditions we're either using a $T$-test or a $Z$-test: 
* We use $T$-test (**independent**, since two of our groups do not intersect. There's also a "paired" type) when:
    * The sample size is small ($n\lt30$)
    * **and/or** the underlying population's variance is unknown
* And we use $Z$-test when:
    * The sample size $n\geq30$
    * **and** the underlying population's $\sigma$ is known and is unlikely to change over time

## Types of T-tests. Calculating statistics
For the Welch $T$-test (we assume that variances are different for $A$ and $B$) we calculate $t_{stat}=\frac{\mu_A-\mu_B}{\sqrt{SE^2_A+SE^2_B}}$, where $SE=\frac{\sigma}{\sqrt{n}}$, which measures the variability around the mean. The bigger the $n$, the closer our estimate shoud be to our real parameter's value.

Sometimes one could also see a $T$-test with variances for both groups assumed to be equal (also referred as a "pooled variance"). In this case a test is referred as as the euql variance $T$-test with $t^{*}_{stat}=\frac{\mu_A-\mu_B}{\sqrt{\sigma_{pooled}^2(\frac{1}{n_A}+\frac{1}{n_B})}}$. Under this assumption $\sigma_{pooled}^2=\frac{(n_A-1)\sigma_A^2+(n_B-1)\sigma_B^2}{n_A+n_B-2}$, which is a weighted average between $\sigma^2_A$ and $\sigma^2_B$. In fact for a $n_A=n_B$ case $\sigma_{pooled}^2=\frac{\sigma_A^2+\sigma_B^2}{2}$. And if we add to it, the fact that $\sigma_A=\sigma_B$, then the formula would align with the original $t_{stat}$ used in Welch test.

Both $t_{stat}$ and $t^{*}_{stat}$ follow a Student's T distribution $T(d)$, which requires a $d$ parameter that is related to degrees of freedom (d.o.f.). While d.o.f. for $d_{t^{*}}=n_A+n_B-2$, the d.o.f. for Welch's test $d_{t}=\frac{(SE^2_A+SE^2_B)^2}{[SE^2_A]^2/(n_A-1)+[SE^2_B]^2/(n_B-1)}$, which has been derived experimentally.

To sum up: 
* when in fact $\sigma_A=\sigma_B$ or we're really sure about it, then the **equal variances t-test** adds up just a litle more power to the AB test. **Power** (referred as $1-\beta$) in hypothesis testing refers to an inverse probability $1-P(\text{type II error})$ (as opposed to $\alpha=P(\text{Type I error})$). **Type II error** refers to failing to reject $H_0$, while $H_1$ is in fact true. What we must remember is that the **higher power** means that we become **more sensistive** in the difference of $\mu$ between two groups. A more visually intuitive explanation is available [here](https://online.stat.psu.edu/stat415/lesson/25/25.1). We can also confirm that if we expand the denominator term in both t-tests. The Welch's test would have made the denominator of the denominator smaller, i.e. ${n_An_B}$ instead of a smaller ${n_A+n_B-2}$. This could result into a higher statistic, higher $\beta$ and, therefore, a lower power, which could negatively influence test's results if there's a significant difference between $A$ and $B$;
* otherwise, **Welch's test** is more appropriate (particularly when sample sizes differ as well)

## A little more on degrees of freedom
Degrees of freedom give you an estimate of the maximumn number of logically independent values, which may vary in a data sample. Consider a t-test with equal variances. That means that only the mean could vary between $A$ and $B$. Let's look at the sample $A$. We are also aware about the sample's mean $\mu_A$, that is made up of all $n_A$ entries in the sample. However, the last entry does not matter, because it is dependent on the mean of all previous entries. We cannot say the same for last two or more entries, because we can still move them around as we want, so that together they all add up to $\mu_A$. That is why our important entries are only limited by $n_A-1$. That is the same case for $B$, hence a total d.o.f. is equal to $n_A+n_B-2$. Degrees of freedom also play a role in creating un unbiased estimate. 

In the case of Student's T distribution d.o.f. labeled as $d$ play a role of an "approximation" of variance for the normal distribution. The larger the $d$, the closer we are to $\mathbb{N}(0, 1)$ and the less is the variance. This happens due to a clever usage of the gamma function $\Gamma(d)=(d-1)!$, which has very useful properties related to exponents.

## Z-test. Calculating statistic
In $Z$-test we calculate a statistic $z_{stat}=\frac{\mu_A-\mu_B}{\sqrt{\sigma_A^2 / n_A+\sigma_B^2 / n_B}}$, where $\sigma_A$ and $\sigma_B$ are assumed to be known. $z_{stat}$ follows $\mathbb{N}(0, 1)$, which is the main difference with Welch's T test.

## Why more emphasis on T-test
It is empirically proven that T-test is a pretty robust framework for samples of arbitrary sizes. When the difference between $A$ and $B$ barely makes it over $\alpha$, it's better to use a T-test, because it gives an approximation of a distribution of means. At the same time with the larger sample size the Central Limit Theorem ensures that our distribution of sample means follows a normal one. This means that the difference between statistics generated by $T$-distribution and $Z$-distribution changes, hence the difference in generated p-values would also become smaller.