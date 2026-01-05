## Bigram Probability Calculation

- Bigrams take a step further by considering previous occuring word to predict the next word.
- The Bigram model assumes that the probability of a word depends only on the word immediately preceding it.
- The probability of a word W under the Bigram model , given the previous word is B:

$P(W)=P(W|B)=\frac{\text{Count}(B, W)}{\text{Count}(B)}$

Where:

- $\text{Count}(B)$ = Number of times the word B appears in the corpus.

- $\text{Count}(B, W)$ = Number of time the word W occurs after the Word B in the corpus.
- Round the answer upto the 4th decimal place.

---

### Sample Corpus

```text
<s> I am Jack </s>
<s> Jack I am </s>
<s> Jack I like </s>
<s> Jack I do like </s>
<s> do I like Jack </s>
```

Notes : 
- \<s> : Start of a sentence
- \</s> : End of a sentence
- Need to count both the start and enod of sentence tokens while calculating probability.
- Zero probability issues are not addressed here and will be covered separately under smoothing techniques in  later problems.
