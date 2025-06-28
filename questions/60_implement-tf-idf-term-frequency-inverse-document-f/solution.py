import numpy as np


def compute_tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents using only NumPy.
    The output TF-IDF scores retain five decimal places.
    """
    vocab = sorted(set(word for document in corpus for word in document).union(query))
    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    tf = np.zeros((len(corpus), len(vocab)))

    for doc_idx, document in enumerate(corpus):
        for word in document:
            word_idx = word_to_index[word]
            tf[doc_idx, word_idx] += 1
        tf[doc_idx, :] /= len(document)

    df = np.count_nonzero(tf > 0, axis=0)

    num_docs = len(corpus)
    idf = np.log((num_docs + 1) / (df + 1)) + 1

    tf_idf = tf * idf

    query_indices = [word_to_index[word] for word in query]
    tf_idf_scores = tf_idf[:, query_indices]

    tf_idf_scores = np.round(tf_idf_scores, 5)

    return tf_idf_scores.tolist()
