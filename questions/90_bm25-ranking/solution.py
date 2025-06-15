import numpy as np
from collections import Counter


def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
    if not corpus or not query:
        raise ValueError("Corpus and query cannot be empty")

    doc_lengths = [len(doc) for doc in corpus]
    avg_doc_length = np.mean(doc_lengths)
    doc_term_counts = [Counter(doc) for doc in corpus]
    doc_freqs = Counter()
    for doc in corpus:
        doc_freqs.update(set(doc))

    scores = np.zeros(len(corpus))
    N = len(corpus)

    for term in query:
        df = doc_freqs.get(term, 0) + 1
        idf = np.log((N + 1) / df)

        for idx, term_counts in enumerate(doc_term_counts):
            if term not in term_counts:
                continue

            tf = term_counts[term]
            doc_len_norm = 1 - b + b * (doc_lengths[idx] / avg_doc_length)
            term_score = (tf * (k1 + 1)) / (tf + k1 * doc_len_norm)
            scores[idx] += idf * term_score

    return np.round(scores, 3)
