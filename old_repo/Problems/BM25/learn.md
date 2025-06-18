## **BM25**

BM25 (Best Match 25) is used in information retrieval for search relevance. Similar to 
TF-IDF is a numerical statistic used to reflect the importance of a word in a document within a collection or corpus.

While TF-IDF is a good metric and is very simple to compute, it has some limitations that BM25 mitigates:

1. **__Saturation__** : Having a single term in the document multiple times skews the Term Frequency factor in TF-IDF.
   While having some terms makes the document relevant, having the same term thousands of times doesn't make the document
   any more relevant and opens up the opportunity for hacking the system. This is mitigated in BM25 by replacing TF by
   TF / (TF + k1)
   
2. **__Document length normalization__** : Another problem not addressed by the TF-IDF is the document length.
   Intuitively, a document of shorter length with more frequency of relevance is higher ranked than a document with the
   same frequency but of longer length. To achieve this we multiply k1 by the ratio dl/adl.
   Where dl = document’s length, adl = average document length across the corpus

3. **__Amplifying Parameter__** : Lastly, we want to parametrize the penalty factor for the document length.
   This is achieved through the variable b. If b is bigger, the effects of the length of the document
   compared to the average length are more amplified.

Final Formula : BM25 = TF/(TF + k1*(1 - b + b*dl/adl)) 

**Implementation Steps**
Compute Document Length and Average Document Length 
Compute TF: For each term in each document, calculate its term frequency using the adapted formula for BM25
Compute IDF: Calculate the inverse document frequency for each unique term in the corpus.
Calculate TF-IDF: Multiply TF and IDF for each term in each document.
Normalize: Normalize the TF-IDF vectors for each document.
