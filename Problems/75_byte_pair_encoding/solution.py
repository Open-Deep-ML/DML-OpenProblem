import re
from collections import Counter

# Step 1: Initialize vocabulary
def initialize_vocab(corpus):
    vocab = Counter()
    for word in corpus.split():
        # Add end-of-word token for clarity
        word = ' '.join(list(word)) + ' </w>'
        vocab[word] += 1
    return vocab

# Step 2: Find symbol pairs and their frequencies
def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

# Step 3: Merge the most frequent pair
def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab:
        # Replace the most frequent pair with a new symbol
        new_word = pattern.sub(''.join(pair), word)
        new_vocab[new_word] = vocab[word]
    return new_vocab

# Step 4: Identify and keep separate tokens in the vocabulary
def get_separate_tokens(vocab):
    separate_tokens = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for symbol in symbols:
            # Count each symbol that is not part of a merged token
            separate_tokens[symbol] += freq
    return separate_tokens

# Step 5: Learn BPE rules with separate tokens
def learn_bpe_with_separate_tokens(corpus, num_merges):
    vocab = initialize_vocab(corpus)
    bpe_rules = []

    print("Initial Vocabulary:")
    print(vocab)

    for merge_step in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break

        # Find the most frequent pair
        most_frequent = max(pairs, key=pairs.get)
        bpe_rules.append(most_frequent)

        # Merge the most frequent pair
        vocab = merge_vocab(most_frequent, vocab)

        print(f"\nStep {merge_step + 1}: Most frequent pair: {most_frequent}")
        print(f"Updated Vocabulary: {vocab}")

    # Extract separate tokens as standalone entries
    separate_tokens = get_separate_tokens(vocab)

    print("\nFinal Separate Tokens:")
    print(separate_tokens)

    return bpe_rules, vocab, separate_tokens

# Example usage
corpus = '''Tokenization is the process of breaking down 
a sequence of text into smaller units called tokens,
which can be words, phrases, or even individual characters.
Tokenization is often the first step in natural languages processing tasks 
such as text classification, named entity recognition, and sentiment analysis.
The resulting tokens are typically used as input to further processing steps,
such as vectorization, where the tokens are converted
into numerical representations for machine learning models to use.'''
num_merges = 20

# Learn BPE rules with detailed steps and separate tokens
bpe_rules, final_vocab, separate_tokens = learn_bpe_with_separate_tokens(corpus, num_merges)

print("\nFinal Vocabulary:")
print(final_vocab)
print("\nSeparate Tokens as Entries:")
print(separate_tokens)
