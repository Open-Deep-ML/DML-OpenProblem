
## Hyperdimensional Computing
Hyperdimensional Computing (HDC) is a computational model inspired by the brain's ability to represent and process information using high-dimensional vectors, based on hypervectors being quasi-orthogonal. It uses vectors with a large number of dimensions to represent data, where each vector is typically filled with binary (1 or 0) or bipolar values (1 or -1). To represent complex data patterns, binding and bundling operations are used. 

In HDC, different data types such as numeric and categorical variables are projected into high-dimensional space through specific encoding processes. Categorical variables are assigned unique hypervectors, often randomly generated binary or bipolar vectors, that serve as representations for each category. Numeric variables are encoded by discretizing the continuous values and mapping discrete bins to hypervectors. These projections allow HDC models to integrate various data types into a unified high-dimensional representation, preserving information across complex, multi-feature datasets.

---

## Binding Operation
The binding operation between two hypervectors is performed element-wise using multiplication. This operation is used to represent associations between different pieces of information:

$$
\text{bind}(\text{hv1}, \text{hv2}) = \text{hv1} \times \text{hv2}
$$

Where $ \text{hv1} $ and $ \text{hv2} $ are bipolar vectors, and their element-wise multiplication results in a new vector where each element is either 1 or -1.

---

## Bundling Operation
The bundling operation sums multiple hypervectors to combine information, typically using element-wise addition for bipolar vectors and XOR operations for binary vectors. This operation aggregates information and creates a composite hypervector that represents the overall data or concept. For example, for a set of $ n $ hypervectors $ \text{hv1}, \text{hv2}, \dots, \text{hvn} $, the bundled vector is:

$$
\text{bundle}(\text{hv1}, \text{hv2}, \dots, \text{hvn}) = \sum_{i=1}^{n} \text{hvi}
$$

This bundled vector is then normalized to ensure it remains bipolar.

---

## Normalization
Normalization ensures that the final bundled vector contains only bipolar or binary values. The normalization function typically applies a thresholding process that transforms any value greater than zero to +1 and any value less than zero to -1. Zero values are then typically assigned to either +1 or -1.

---

## Operations in Practice: Example
Consider a scenario where we want to represent and combine information from each feature in a row of a dataset. Each feature, whether numeric or categorical, is represented by a hypervector, and these hypervectors are combined to form a composite vector that represents the entire row of data.

For instance, if we have a dataset row with features Feature A and Feature B, we would:
1. Create a hypervector for the column Feature A and another for its specific feature value.
2. Create a hypervector for the column Feature B and another for its specific feature value.
3. Bind each feature’s column hypervector with the hypervector representing its value to form a unique vector for each feature.
4. Bundle all the feature hypervectors for this row to create a single composite vector representing the entire row.
5. Normalize the bundled vector to maintain bipolar values.

---

## Applications of HDC
Hyperdimensional computing has a variety of applications, including:
1. **Data Classification**: Using high-dimensional vectors to represent data points and classifying them based on their properties.
2. **Pattern Recognition**: Recognizing complex patterns in data through binding and bundling operations.
3. **Natural Language Processing**: Representing words and phrases as high-dimensional vectors to analyze and process text data.
