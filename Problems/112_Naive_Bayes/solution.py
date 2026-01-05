import re
import math
from collections import defaultdict

def preprocess_text(text):
    """Convert text to lowercase and remove non-alphabetic characters."""
    return re.findall(r'\b[a-z]+\b', text.lower())

def train_naive_bayes(data):
    """Train Naïve Bayes on labeled text data."""
    word_counts = {"spam": defaultdict(int), "ham": defaultdict(int)}
    class_counts = {"spam": 0, "ham": 0}
    total_words = {"spam": 0, "ham": 0}

    for text, label in data:
        words = preprocess_text(text)
        class_counts[label] += 1
        total_words[label] += len(words)
        for word in words:
            word_counts[label][word] += 1

    return word_counts, class_counts, total_words

def calculate_probabilities(word_counts, class_counts, total_words, text):
    """Calculate the probability of a text being spam or ham."""
    words = preprocess_text(text)
    total_documents = sum(class_counts.values())

    probs = {}
    for label in ["spam", "ham"]:
        prior = class_counts[label] / total_documents
        log_prob = math.log(prior)

        for word in words:
            word_frequency = word_counts[label].get(word, 0) + 1
            log_prob += math.log(word_frequency / (total_words[label] + len(word_counts[label])))

        probs[label] = log_prob

    return "spam" if probs["spam"] > probs["ham"] else "ham"

def test_naive_bayes():
    """Run test cases to verify Naïve Bayes implementation."""
    data = [
        ("free money now", "spam"),
        ("win lottery today", "spam"),
        ("hello friend", "ham"),
        ("meeting at five", "ham"),
        ("urgent cash prize", "spam"),
        ("call me later", "ham")
    ]

    word_counts, class_counts, total_words = train_naive_bayes(data)

    test_cases = [
        ("win cash now", "spam"),
        ("hello there", "ham"),
        ("lottery prize win", "spam"),
        ("meeting at noon", "ham")
    ]

    for text, expected in test_cases:
        prediction = calculate_probabilities(word_counts, class_counts, total_words, text)
        assert prediction == expected, f"Failed on '{text}': Expected {expected}, got {prediction}"

    print("All Naïve Bayes tests passed.")

if __name__ == "__main__":
    test_naive_bayes()
