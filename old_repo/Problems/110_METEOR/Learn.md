METEOR(Metric for Evaluation of Translation with Explicit ORdering) is a metric generally used for 
machine translation and evaluating the text output of generative AI models. METEOR build was introduced to address 
the limitations in earlier metrics like BLEU.

## Key Characteristics
- Considers semantic similarity beyond exact word matching
- Accounts for word order and translation variations
- Provides more human-aligned translation assessment

# Implementation 
1. **Tokenization**

2. **Frequency of matching words** : Matching needs to be exact

3. **Calculate Precision, Recall and F-mean**
```
   F_mean = (Precision * Recall) / (α * Precision + (1 - α) * Recall)
```
   - α typically set to 0.9
   - Balances precision and recall

4. **Fragmentation Penalty**
   ```
   Chunks = Count of contiguous matched word sequences
   Penalty = γ * (Chunks / Matches)^β
   ```
   - β controls penalty weight (typically 3)
   - γ limits maximum penalty (typically 0.5)

5. **Final METEOR Score**
   ```
   METEOR = F_mean * (1 - Penalty)
   ```
   - Ranges from 0 (no match) to 1 (perfect match)

**__Note__** : The [paper](https://aclanthology.org/W05-0909/) that introduced the metric doesn't have the parameters (α,β, and γ) as tunable parameters, but implementation in other libraries like NLTK offers this flexibility.

# Example 

- Reference: "The quick brown fox jumps over the lazy dog"
- Candidate: "A quick brown fox jumps over a lazy dog"

### 1. Tokenization
- Reference Tokens: ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
- Candidate Tokens: ['a', 'quick', 'brown', 'fox', 'jumps', 'over', 'a', 'lazy', 'dog']

### 2. Unigram Matching
- Matching tokens: ['quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']
- Matches: 7

### 3. Unigram Precision and Recall Calculation
- Precision = Matches / Candidate Length = 7 / 9 ≈ 0.778

- Recall = Matches / Reference Length = 7 / 9 ≈ 0.778

### 4. F-mean Calculation (α = 0.9)
```
F_mean = (Precision * Recall) / (α * Precision + (1 - α) * Recall)
       = (0.778 * 0.778) / (0.9 * 0.778 + (1 - 0.9) * 0.778)
       = 0.606 / (0.7 + 0.078)
       = 0.606 / 0.778
       ≈ 0.779
```

### 5. Chunk Calculation
- Contiguous matched sequences:
  1. ['quick', 'brown', 'fox']
  2. ['jumps', 'over']
  3. ['lazy', 'dog']
- Number of Chunks: 3
- Total Number of Unigram Matches: 7

### 6. Penalty Calculation (β = 3, γ = 0.5)
```
Penalty = γ * (Number of Chunks / Total Number of Unigram Matches)^β
        = 0.5 * (3 / 7)^3
        = 0.5 * (0.429)^3
        ≈ 0.039
```

### 7. Final METEOR Score
```
METEOR = F_mean * (1 - Penalty)
       = 0.779 * (1 - 0.039)
       = 0.779 * 0.961
       ≈ 0.749
```
