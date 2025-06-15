# Pointwise Mutual Information (PMI)

Pointwise Mutual Information (PMI) is a statistical measure used in information theory and Natural Language Processing (NLP) to quantify the association between two events. It measures how much the actual joint occurrence of two events differs from what would be expected if they were independent. PMI is commonly used for identifying word associations, feature selection in text classification, and calculating document similarity.

## Implementation 

1. **Collect Count Data** for events $x$, $y$, and their joint occurrence $(x,y)$.

2. **Calculate Individual Probabilities**:
   - $P(x) = \frac{\text{Count}(x)}{\text{Total Count}}$
   - $P(y) = \frac{\text{Count}(y)}{\text{Total Count}}$

3. **Calculate Joint Probability**:
   - $P(x,y) = \frac{\text{Count}(x,y)}{\text{Total Count}}$

4. **Calculate PMI**:
   - $$\text{PMI}(x,y) = \log_2\left(\frac{P(x,y)}{P(x) \cdot P(y)}\right)$$

## Interpretation of PMI Values

- **Positive PMI**: Events co-occur more frequently than expected by chance.
- **Zero PMI**: Events are statistically independent.
- **Negative PMI**: Events co-occur less frequently than expected by chance.
- **Undefined PMI**: Occurs when $P(x,y) = 0$ (the events never co-occur).

## Variants of PMI

### 1. Normalized PMI (NPMI)

NPMI scales PMI to a range of [-1, 1] to account for dataset size variations:

$$
\text{NPMI}(x,y) = \frac{\text{PMI}(x,y)}{-\log_2 P(x,y)}
$$

### 2. Positive PMI (PPMI)

PPMI sets negative PMI scores to zero, often used in word embeddings:

$$
\text{PPMI}(x,y) = \max(\text{PMI}(x,y),\,0)
$$
