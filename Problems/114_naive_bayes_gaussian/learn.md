# **Naive Bayes Classifier**

## **1. Definition**

Naive Bayes is a **probabilistic machine learning algorithm** used for **classification tasks**. It is based on **Bayes' Theorem**, which describes the probability of an event based on prior knowledge of related events.

The algorithm assumes that:
- **Features are conditionally independent** given the class label (the "naive" assumption).
- It calculates the posterior probability for each class and assigns the class with the **highest posterior** to the sample.

---

## **2. Bayes' Theorem**

Bayes' Theorem is given by:

\[
P(C | X) = \frac{P(X | C) \times P(C)}{P(X)}
\]

Where:
- \( P(C | X) \) → **Posterior** probability: the probability of class \( C \) given the feature vector \( X \)
- \( P(X | C) \) → **Likelihood**: the probability of the data \( X \) given the class
- \( P(C) \) → **Prior** probability: the initial probability of class \( C \) before observing any data
- \( P(X) \) → **Evidence**: the total probability of the data across all classes (acts as a normalizing constant)

Since \( P(X) \) is the same for all classes during comparison, it can be ignored, simplifying the formula to:

\[
P(C | X) \propto P(X | C) \times P(C)
\]

---


### 3 **Gaussian Naive Bayes**
- Assumes the **features follow a normal distribution**.
- For each feature \( x_i \), the likelihood is given by:

\[
P(x_i | C) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x_i - \mu)^2}{2\sigma^2}}
\]

Where:
- \( \mu \) → Mean of the feature for class \( C \)
- \( \sigma^2 \) → Variance of the feature for class \( C \)

---

## **4. Applications of Naive Bayes**

- **Text Classification:** Spam detection, sentiment analysis, and news categorization.
- **Document Categorization:** Sorting documents by topic.
- **Fraud Detection:** Identifying fraudulent transactions or behaviors.
- **Recommender Systems:** Classifying users into preference groups.

---
