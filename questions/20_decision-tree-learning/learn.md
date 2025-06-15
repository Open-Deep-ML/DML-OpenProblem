
## Decision Tree Learning Algorithm

The decision tree learning algorithm is a method used for classification that predicts the value of a target variable based on several input variables. Each internal node of the tree corresponds to an input variable, and each leaf node corresponds to a class label.

### Algorithm Overview
The recursive binary splitting starts by selecting the attribute that best separates the examples according to the entropy and information gain, calculated as follows:

### Entropy
$$
H(X) = -\sum p(x) \log_2 p(x)
$$

### Information Gain
$$
IG(D, A) = H(D) - \sum \frac{|D_v|}{|D|} H(D_v)
$$

### Explanation of Terms
- **Entropy \( H(X) \)**: Measures the impurity or disorder of the set.
- **Information Gain \( IG(D, A) \)**: Represents the reduction in entropy after splitting the dataset \( D \) on attribute \( A \).
- **\( D_v \)**: The subset of \( D \) for which attribute \( A \) has value \( v \).

### Process
1. **Select Attribute**: Choose the attribute with the highest information gain.
2. **Split Dataset**: Divide the dataset based on the values of the selected attribute.
3. **Recursion**: Repeat the process for each subset until:
   - All data is perfectly classified, or
   - No remaining attributes can be used to make a split.

This recursive process continues until the decision tree can no longer be split further or all examples have been classified.
