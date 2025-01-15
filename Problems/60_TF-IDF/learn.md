# Understanding TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF is a numerical statistic that reflects how important a word is in a document relative to a collection (or corpus). It is widely used in information retrieval, text mining, and natural language processing tasks.

## Mathematical Formulation

TF-IDF is the product of two key statistics: **Term Frequency (TF)** and **Inverse Document Frequency (IDF)**.

### 1. Term Frequency (TF)

The term frequency is defined as:

$TF(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$

- $t$: A specific term (word).
- $d$: A specific document in the corpus.

### 2. Inverse Document Frequency (IDF)

To account for how common or rare a term is across all documents in the corpus, we calculate:

$IDF(t) = \log\Bigl(\frac{N + 1}{\text{df}(t) + 1}\Bigr) + 1$

Where:

- $N$: Total number of documents in the corpus.
- $\text{df}(t)$: Number of documents containing the term $t$.
- Adding $+1$ inside the fraction prevents division by zero if a term never appears.
- Adding $+1$ outside the log ensures IDF remains nonzero.

### 3. TF-IDF

Combining TF and IDF:

$TFIDF(t, d) = TF(t, d) \times IDF(t)$

## Implementation Steps

1. **Compute TF**  
   For each document, count how often each term appears and divide by the document’s total word count.

2. **Compute IDF**  
   For each term, calculate its document frequency across all documents and apply the IDF formula.

3. **Calculate TF-IDF**  
   For every term in every document, multiply the term’s TF by its IDF.

4. **Normalization (Optional)**  
   Normalize TF-IDF vectors (e.g., using $L2$ norm) if comparing documents in a vector space model.

## Example Calculation

Suppose we have a small corpus of 3 documents:

- **Doc1**: "The cat sat on the mat"  
- **Doc2**: "The dog chased the cat"  
- **Doc3**: "The bird flew over the mat"

We want to calculate the TF-IDF for the word **"cat"** in **Doc1**.

### Step 1: Compute $TF("cat", \text{Doc1})$

$TF("cat", \text{Doc1}) = \frac{1}{6} \approx 0.1667$

- "cat" appears once.
- Total words in Doc1 (counting each occurrence of “the”) = 6.

### Step 2: Compute $IDF("cat")$

- "cat" appears in 2 out of 3 documents, so $\text{df}("cat") = 2$.
- $N = 3$.

Using the formula with smoothing and an added constant:

$IDF("cat") = \log\Bigl(\frac{N + 1}{\text{df}("cat") + 1}\Bigr) + 1 = \log\Bigl(\frac{3 + 1}{2 + 1}\Bigr) + 1 = \log\Bigl(\frac{4}{3}\Bigr) + 1 \approx 0.2877 + 1 = 1.2877$

### Step 3: Calculate $TFIDF("cat", \text{Doc1})$

$TFIDF("cat", \text{Doc1}) = TF("cat", \text{Doc1}) \times IDF("cat") = 0.1667 \times 1.2877 \approx 0.2147$

## Applications of TF-IDF

1. **Information Retrieval**  
   TF-IDF is often used in search engines to rank how relevant a document is to a given query.
2. **Text Mining**  
   Helps identify key terms and topics in large volumes of text.
3. **Document Classification**  
   Useful for weighting important words in classification tasks.
4. **Search Engines**  
   Refines document ranking by emphasizing distinctive terms.
5. **Recommendation Systems**  
   Evaluates text-based similarity (e.g., for content-based filtering).

TF-IDF remains a foundational technique in natural language processing, widely used for feature extraction and analysis across numerous text-based applications.
