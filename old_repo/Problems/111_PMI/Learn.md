# Pointwise Mutual Information (PMI)

Pointwise Mutual Information (PMI) is a statistical measure used in information theory and Natural Language Processing (NLP) to quantify the level of association between two events. It compares the probability of two events 
occurring together versus the probability of them occurring independently. It is commonly used in Natural Language Processing(NLP) and Information Retrieval to find association between two words, feature selection in text classification, 
document similarity.

## Implementation 
1. **Collect Count Data for event x, y and joint occurence**
   
2. **Calculate Individual Probabilities**

3. **Calculate Joint Probability**

4. **Final Score : PMI(x,y) = log₂(P(x,y) / (P(x) * P(y)))**
   Where:
  - P(x,y) is the probability of events x and y occurring together
    
  - P(x) is the probability of event x occurring
    
  - P(y) is the probability of event y occurring

## Interpreting PMI Values

- **Positive PMI**: Events co-occur more than expected by chance
- **Zero PMI**: Events are statistically independent
- **Negative PMI**: Events co-occur less than expected by chance
- **Undefined**: When P(x,y) = 0 (events never co-occur)

## Variants

### 1. Normalized PMI (NPMI)
- Scales PMI to range [-1, 1]
- Easier to compare across different datasets
- Formula: NPMI(x,y) = PMI(x,y) / -log₂(P(x,y))

### 2. Positive PMI (PPMI)
- Sets negative PMI values to zero
- Commonly used in word embedding models
- Formula: PPMI(x,y) = max(PMI(x,y), 0)
