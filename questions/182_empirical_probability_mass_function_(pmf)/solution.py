from collections import Counter

def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    samples = list(samples)
    if not samples:
        return []
    total = len(samples)
    cnt = Counter(samples)
    result = [(k, cnt[k] / total) for k in sorted(cnt.keys())]
    return result