## Overview
ROC-AUC is a metric used for measuring predictive quality of a binary classifier with the highest value being $1$ and the lowest being $0$.

## $TPR$ and $FPR$
Consider a trivial case, when we have true binary labels $y_i\in\{0, 1\}$ and our predicted labels by the model $\hat{y_i}\in\{0, 1\}$. We also denote any arbitrary example labeled as $1$ as positive and $0$ as negative. Using $(y_i, \hat{y_i})$ combinations we build a set of $Y$, with the help of which we then generate statistics such as $TP$ (True Positive), $TN$ (True Negative), $FP$ (False Positive) and $FN$ (False Negative):

| Total population = P + N     | Predicted positive (PP)                            | Predicted negative (PN)                               |
|-----------------------------------|----------------------------------------------------|-------------------------------------------------------|
| **Actual positive (P)**           | $TP=\#\{Y\|y_i=\hat{y_i}=1\}$                        | $FN=\#\{Y\|y_i=1;\hat{y_i}=0\}$ (also called type II error) |
| **Actual negative (N)**           | $FP=\#\{Y\|y_i=0;\hat{y_i}=1\}$ (also called type I error)                      | $TN=\#\{Y\|y_i=\hat{y_i}=0\}$                           |

This table, also referenced as **confusion matrix**, could provide an overview of the model's performance for this particular task. Now with the help of these statistics we can calculate the following estimates:
$$
TPR=\frac{TP}{TP+FN}\quad(\text{also called a recall}) \\ 

FPR=\frac{FP}{FP+TN}
$$

Intuition-wise, **TPR** shows the model's sensitivity to positive cases, where the true label $y_i=1$. In some cases, for example in credit scoring or cancer detection tasks, we even neglect other metrics in favor of recall, since any $FN$-case could turn out a very costly mistake. **FPR**, on the other hand, shows how biased are we towards positive cases at the expense of $y_i=0$. 

## Thershold
Now recall that we originally obtain a vector $\hat{y_i}\in\{0, 1\}$ of predicted labels. But the model itself is not able to directly output either $0$ or $1$. Instead we look at the probability $z_i$ the model has provided us with and compare it with empirically chosen threshold $t$. For example, for a chosen $t=0.7$ we would have the following decision rule:
$$
\hat{y_i}=\begin{cases} 1, & \text{if } z_i\gt 0.7 \\ 0, & \text{otherwise } \end{cases}
$$

With this idea in mind, we can see that for every $t$ our previous estimates of $TPR$ and $FPR$ would change as well, so we can actually denote them as $TPR(t)$ and $FPR(t)$. But we also want our model to be robust and not be dependent on what thershold we choose. That is why when we need to measure the quality of our model, we often look at the **ROC** curve $TPR(FPR | t)$, which shows $TPR$ and $FPR$ under various thresholds. 

## ROC curve
Each point of this curve is obtained via this algorithm:
$$


\begin{array}{l}
\textbf{Input}: y\_true, y\_pred \text{ (true labels and output probabilities)} \\
\textbf{Output: } \text{points} \text{ (a set of (x, y) coordinates)} \\
\text{\textbf{function} roc\_points}(y\_true, y\_pred): \\
\quad thresholds \leftarrow y\_pred \cup \{0\} \\
\quad points \leftarrow [\quad ] \\
\quad \textbf{for } t\in\{thresholds\,:\ t_i\ge t_{i+1}\} \textbf{ do}: \\
\quad \quad y \leftarrow TPR(t) \\
\quad \quad x \leftarrow FPR(t) \\
\quad \quad \text{points.append}((x, y)) \\
\quad \textbf{end for} \\
\textbf{end function}
\end{array}
$$

ROC curve's domain stays within $[0, 1]$. To break it down, first consider a thershold $t=1$. Then it is impossible to assign any label to our predictions, unless it is $0$. Therefore $TP=0\implies TPR=0$ and $FP=0\implies FPR=0$ (since all negative examples are going to be assigned a correct label). On the other hand if we have $t=0$, then $FN = 0\implies TPR=\frac{TP}{TP}=1$ and $TN = 0 \implies FPR=\frac{FP}{FP}=1$, since there is no way we can assign $0$ to any prediction.

The best case cenario is when with increasing thershold $t$ our sensitivity increases without disregarding the bias ($FPR$ does not change or is around 0 and $TPR$ is always high). The worst case cenario is when the model is random and we follow an $FPR=TPR$ diagonal line. 

## ROC-AUC
If you consider two ROC curves mentioned above, you could see that the space underneath the first one is greater than the second one. This is why we usually calculate **ROC-AUC** - area under the ROC curve. You might think that the larger is the AUC, the better is the model, but in fact it's a common misconception.

Consider you want to choose a model between model #1 with $AUC_{ROC}=0.6$ and model #2 with $AUC_{ROC}=0.3$. The correct answer is actually #2, since we can always invert our decision rule in favor of the ROC-AUC and our $AUC_{ROC}$ for model #2 would actually become $0.7$. Therefore, when looking at the ROC AUC, we should consider how large is the **absolute** difference between the area of $0.5$ (worst case performance) and the one our model has generated.

## Calculating AUC
There are also various ways for calculating an area under the curve. The most applicable one, which is also used in scikit-learn, is the trapezoidal rule:
$$
\int f(x)=\sum_i\frac{1}{2}\Delta x_i * (f(x_i)-f(x_{i-1})) ,
$$

where $\Delta x_i=x_i-x_{i-1}$. This method breaks a total area under the curve into a sum of $90^\circ$-rotated trapezoids that make up the convex curve.