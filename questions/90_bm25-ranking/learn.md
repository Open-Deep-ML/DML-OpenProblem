## **BM25**

BM25 (Best Match 25) is used in information retrieval for search relevance. Similar to TF-IDF, it reflects the importance of a word in a document within a collection or corpus. However, BM25 improves upon TF-IDF by addressing key limitations.

### Limitations of TF-IDF Addressed by BM25

1. **Saturation**: In TF-IDF, having a term multiple times in a document skews the term frequency, making the document overly relevant. BM25 mitigates this by using:  
   $$ \text{TF-adjusted} = \frac{\text{TF}}{\text{TF} + k_1} $$  

2. **Document Length Normalization**: BM25 accounts for document length by normalizing term frequencies using:  
   $$ \text{Normalized Length} = 1 - b + b \times \frac{\text{Doc Len}}{\text{Average Doc Len}} $$  

3. **Amplifying Parameter**: The $b$ parameter controls the influence of document length normalization. Higher $b$ values amplify the effect.

### Final BM25 Formula

The BM25 score for a term is given by:  
$$ \text{BM25} = \text{IDF} \times \frac{\text{TF} \times (k_1 + 1)} {\text{TF} + k_1 \times (1 - b + b \times \frac{\text{dl}}{\text{adl}})} $$  

Where:  
- $ \text{TF} $: Term frequency in the document.  
- $ \text{IDF} $: Inverse document frequency, calculated as $ \log(\frac{N + 1}{\text{df} + 1}) $.  
- $ N $: Total number of documents.  
- $ \text{df} $: Number of documents containing the term.  
- $ \text{dl} $: Document length.  
- $ \text{adl} $: Average document length.  
- $ k_1 $: Saturation parameter.  
- $ b $: Normalization parameter.

### Implementation Steps

1. Compute document length ($ dl $) and average document length ($ adl $).  
2. Calculate term frequencies ($ TF $) using the BM25 formula.  
3. Compute inverse document frequencies ($ IDF $) for each term.  
4. Calculate BM25 scores for each document.

### Applications

BM25 is widely used in:  
- Search Engines  
- Recommendation Systems  
- Natural Language Processing (NLP)  

Understanding BM25 enables the creation of robust systems for search and ranking tasks.
