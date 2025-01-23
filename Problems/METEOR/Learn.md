METEOR(Metric for Evaluation of Translation with Explicit ORdering) is a metric generally used for 
machine translation and evaluating the text output of generative AI models. METEOR build was introduced to addresses 
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
   Penalty = min(γ, 0.5 * (Chunks / Matches)^β)
   ```
   - β controls penalty weight (typically 3)
   - γ limits maximum penalty (typically 0.5)

5. **Final METEOR Score**
   ```
   METEOR = F_mean * (1 - Penalty)
   ```
   - Ranges from 0 (no match) to 1 (perfect match)
