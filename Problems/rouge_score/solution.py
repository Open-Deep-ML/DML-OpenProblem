import numpy as np
from collections import Counter
from typing import Dict

def get_ngrams(text: str, n: int) -> Counter:
    """
    Convert text into n-grams and return their counts.
    
    Args:
        text (str): Input text
        n (int): Size of n-grams
    
    Returns:
        Counter: Dictionary with n-grams as keys and their counts as values
    """
    words = text.lower().split()
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return Counter(ngrams)

def rouge_n(reference: str, candidate: str, n: int = 1) -> Dict[str, float]:
    """
    Calculate ROUGE-N score between reference and candidate texts.
    
    Args:
        reference (str): Reference text
        candidate (str): Candidate text to evaluate
        n (int): Size of n-grams to consider (default: 1)
    
    Returns:
        dict: Dictionary containing precision, recall, and f1-score
    """
    # Get n-gram counts
    ref_ngrams = get_ngrams(reference, n)
    cand_ngrams = get_ngrams(candidate, n)
    
    # Convert to numpy arrays for faster computation
    all_ngrams = list(set(ref_ngrams) | set(cand_ngrams))
    ref_vec = np.array([ref_ngrams[ng] for ng in all_ngrams])
    cand_vec = np.array([cand_ngrams[ng] for ng in all_ngrams])
    
    # Calculate overlap using element-wise minimum
    overlap = np.minimum(ref_vec, cand_vec).sum()
    
    # Calculate precision and recall
    precision = overlap / max(cand_vec.sum(), 1e-10)  # Avoid division by zero
    recall = overlap / max(ref_vec.sum(), 1e-10)
    
    # Calculate F1 score
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }