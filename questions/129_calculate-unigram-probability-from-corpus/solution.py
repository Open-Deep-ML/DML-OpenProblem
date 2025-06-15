def unigram_probability(corpus: str, word: str) -> float:
    tokens = corpus.split()
    total_word_count = len(tokens)
    word_count = tokens.count(word)
    return round(word_count / total_word_count, 4)
