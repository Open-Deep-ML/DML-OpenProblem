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

#Example usage (3 tests)
corpus = '''Tokenization is the process of breaking down 
a sequence of text into smaller units called tokens,
which can be words, phrases, or even individual characters.
Tokenization is often the first step in natural languages processing tasks 
such as text classification, named entity recognition, and sentiment analysis.
The resulting tokens are typically used as input to further processing steps,
such as vectorization, where the tokens are converted
into numerical representations for machine learning models to use.'''
num_merges = 20
'''
corpus = "low lower lowest"
num_merges = 10
Answer: Counter({'low</w>': 1, 'lower</w>': 1, 'lowest</w>': 1})
Explanation: 
Initial Vocabulary:
Counter({'l o w </w>': 1, 'l o w e r </w>': 1, 'l o w e s t </w>': 1})

Step 1: Most frequent pair: ('l', 'o')
Updated Vocabulary: {'lo w </w>': 1, 'lo w e r </w>': 1, 'lo w e s t </w>': 1}

Step 2: Most frequent pair: ('lo', 'w')
Updated Vocabulary: {'low </w>': 1, 'low e r </w>': 1, 'low e s t </w>': 1}

Step 3: Most frequent pair: ('low', 'e')
Updated Vocabulary: {'low </w>': 1, 'lowe r </w>': 1, 'lowe s t </w>': 1}

Step 4: Most frequent pair: ('low', '</w>')
Updated Vocabulary: {'low</w>': 1, 'lowe r </w>': 1, 'lowe s t </w>': 1}

Step 5: Most frequent pair: ('lowe', 'r')
Updated Vocabulary: {'low</w>': 1, 'lower </w>': 1, 'lowe s t </w>': 1}

Step 6: Most frequent pair: ('lower', '</w>')
Updated Vocabulary: {'low</w>': 1, 'lower</w>': 1, 'lowe s t </w>': 1}

Step 7: Most frequent pair: ('lowe', 's')
Updated Vocabulary: {'low</w>': 1, 'lower</w>': 1, 'lowes t </w>': 1}

Step 8: Most frequent pair: ('lowes', 't')
Updated Vocabulary: {'low</w>': 1, 'lower</w>': 1, 'lowest </w>': 1}

Step 9: Most frequent pair: ('lowest', '</w>')
Updated Vocabulary: {'low</w>': 1, 'lower</w>': 1, 'lowest</w>': 1}

Final Separate Tokens:
Counter({'low</w>': 1, 'lower</w>': 1, 'lowest</w>': 1})
'''

'''
corpus = "eat sleep code repeat"
num_merges = 10
Answer: Counter({'eat</w>': 2, 'sleep</w>': 1, 'cod': 1, 'e': 1, '</w>': 1, 'r': 1, 'ep': 1})
Explanation:
Initial Vocabulary:
Counter({'e a t </w>': 1, 's l e e p </w>': 1, 'c o d e </w>': 1, 'r e p e a t </w>': 1})

Step 1: Most frequent pair: ('e', 'a')
Updated Vocabulary: {'ea t </w>': 1, 's l e e p </w>': 1, 'c o d e </w>': 1, 'r e p ea t </w>': 1}

Step 2: Most frequent pair: ('ea', 't')
Updated Vocabulary: {'eat </w>': 1, 's l e e p </w>': 1, 'c o d e </w>': 1, 'r e p eat </w>': 1}

Step 3: Most frequent pair: ('eat', '</w>')
Updated Vocabulary: {'eat</w>': 1, 's l e e p </w>': 1, 'c o d e </w>': 1, 'r e p eat</w>': 1}

Step 4: Most frequent pair: ('e', 'p')
Updated Vocabulary: {'eat</w>': 1, 's l e ep </w>': 1, 'c o d e </w>': 1, 'r ep eat</w>': 1}

Step 5: Most frequent pair: ('s', 'l')
Updated Vocabulary: {'eat</w>': 1, 'sl e ep </w>': 1, 'c o d e </w>': 1, 'r ep eat</w>': 1}

Step 6: Most frequent pair: ('sl', 'e')
Updated Vocabulary: {'eat</w>': 1, 'sle ep </w>': 1, 'c o d e </w>': 1, 'r ep eat</w>': 1}

Step 7: Most frequent pair: ('sle', 'ep')
Updated Vocabulary: {'eat</w>': 1, 'sleep </w>': 1, 'c o d e </w>': 1, 'r ep eat</w>': 1}

Step 8: Most frequent pair: ('sleep', '</w>')
Updated Vocabulary: {'eat</w>': 1, 'sleep</w>': 1, 'c o d e </w>': 1, 'r ep eat</w>': 1}

Step 9: Most frequent pair: ('c', 'o')
Updated Vocabulary: {'eat</w>': 1, 'sleep</w>': 1, 'co d e </w>': 1, 'r ep eat</w>': 1}

Step 10: Most frequent pair: ('co', 'd')
Updated Vocabulary: {'eat</w>': 1, 'sleep</w>': 1, 'cod e </w>': 1, 'r ep eat</w>': 1}

Final Separate Tokens:
Counter({'eat</w>': 2, 'sleep</w>': 1, 'cod': 1, 'e': 1, '</w>': 1, 'r': 1, 'ep': 1})

'''

# Learn BPE rules with detailed steps and separate tokens
bpe_rules, final_vocab, separate_tokens = learn_bpe_with_separate_tokens(corpus, num_merges)

print("\nFinal Vocabulary:")
print(final_vocab)
print("\nSeparate Tokens as Entries:")
print(separate_tokens)
