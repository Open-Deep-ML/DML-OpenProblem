# Min-Max Normalization

Min-Max Normalization is a technique used to scale numerical data between *0 and 1*. It ensures that values are transformed while maintaining their original distribution. A machine learning model usually assigns a higher weight to a feature with larger values and a lower weight to a feature with smaller values.  
The goal of normalization is to make every datapoint have the same scale so each feature is equally important to the model.


The formula used for Min-Max Normalization is:

$$
X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
$$

- Note 1 : In case the input array has identical elements just return an array of the same size filled with zeros.
- Note 2 : Round the resultant values upto the 4th decimal place.

## Where:
- \( X \) is the original value.
- $$(X_{\min})$$ is the minimum value in the dataset.
- $$( X_{\text{max}} )$$ is the maximum value in the dataset.
- \( X' \) is the normalized value.
