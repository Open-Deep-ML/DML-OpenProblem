import numpy as np

def compute_tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents using only NumPy.
    The output TF-IDF scores retain five decimal places.
    
    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: List of lists containing TF-IDF scores for the query words in each document,
             with scores rounded to five decimal places
    """
    # Build the vocabulary from the corpus and include query terms
    vocab = sorted(set(word for document in corpus for word in document).union(query))
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    
    # Initialize term-frequency (TF) matrix
    tf = np.zeros((len(corpus), len(vocab)))
    
    # Compute term frequencies
    for doc_idx, document in enumerate(corpus):
        for word in document:
            word_idx = word_to_index[word]
            tf[doc_idx, word_idx] += 1
        # Normalize TF values by the document length
        tf[doc_idx, :] /= len(document)
    
    # Compute document frequency (DF) for each term
    df = np.count_nonzero(tf > 0, axis=0)
    
    # Compute inverse document frequency (IDF) with smoothing
    num_docs = len(corpus)
    idf = np.log((num_docs + 1) / (df + 1)) + 1  # Add 1 to numerator and denominator to prevent division by zero
    
    # Compute TF-IDF matrix
    tf_idf = tf * idf
    
    # Extract TF-IDF scores for the query words
    query_indices = [word_to_index[word] for word in query]
    tf_idf_scores = tf_idf[:, query_indices]
    
    # Round the TF-IDF scores to five decimal places
    tf_idf_scores = np.round(tf_idf_scores, 5)
    
    # Convert the TF-IDF scores to a list of lists
    tf_idf_scores_list = tf_idf_scores.tolist()
    
    return tf_idf_scores_list

def test_tf_idf():
    # Test case 1: Simple corpus with single-word query
    corpus_1 = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "chased", "the", "cat"],
        ["the", "bird", "flew", "over", "the", "mat"]
    ]
    query_1 = ["cat"]
    expected_output_1 = [[0.21461], [0.25754], [0.0]]
    output_1 = compute_tf_idf(corpus_1, query_1)
    assert np.allclose(output_1, expected_output_1, atol=1e-5), \
        f"Test case 1 failed: expected {expected_output_1}, got {output_1}"
    
    # Test case 2: Simple corpus with multi-word query
    corpus_2 = corpus_1  # Reuse the corpus from test case 1
    query_2 = ["cat", "mat"]
    expected_output_2 = [[0.21461, 0.21461], [0.25754, 0.0], [0.0, 0.21461]]
    output_2 = compute_tf_idf(corpus_2, query_2)
    assert np.allclose(output_2, expected_output_2, atol=1e-5), \
        f"Test case 2 failed: expected {expected_output_2}, got {output_2}"
    
    # Test case 3: Larger corpus with multi-word query
    corpus_3 = [
        ["this", "is", "a", "sample"],
        ["this", "is", "another", "example"],
        ["yet", "another", "sample", "document"],
        ["one", "more", "document", "for", "testing"]
    ]
    query_3 = ["sample", "document", "test"]
    expected_output_3 = [
        [0.37771, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.37771, 0.37771, 0.0],
        [0.0, 0.30217, 0.0]
    ]
    output_3 = compute_tf_idf(corpus_3, query_3)
    assert np.allclose(output_3, expected_output_3, atol=1e-5), \
        f"Test case 3 failed: expected {expected_output_3}, got {output_3}"
    
    print("All TF-IDF tests passed.")

if __name__ == "__main__":
    test_tf_idf()
