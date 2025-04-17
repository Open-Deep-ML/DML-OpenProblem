# Unigram Probability Calculation

- In Natural Language Processing (NLP), a  unigram model is the simplest form of a language model. 
- It assumes each word in a sentence is generated independently.  


- The probability of a word w under the unigram model is:

$P(w) = \frac{\text{Count}(w)}{\sum_{w' \in V} \text{Count}(w')}$

Where:

- $\text{Count}(w)$ = Number of times the word w appears in the corpus.

- $V$ = Vocabulary (all word tokens in the corpus).

- $\sum_{w' \in V} \text{Count}(w')$ = Total number of word tokens.
- Round upto the 4th decimal point.


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
