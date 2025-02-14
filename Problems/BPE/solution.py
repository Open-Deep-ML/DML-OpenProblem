class BytePairEncoder:
    def __init__(self):
        self.mappings = {}
        self.next_symbol = 0

    def find_most_frequent_pair(self, text):
        """
        Find the most frequent consecutive pair of characters in the text.
        If multiple pairs have same frequency, return the first occurring pair.
        """
        if len(text) < 2:
            return None

        # Count frequencies of all pairs
        pair_counts = {}
        first_occurrences = {}

        for i in range(len(text) - 1):
            pair = text[i : i + 2]
            if pair not in first_occurrences:
                first_occurrences[pair] = i
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            return None

        # Find the most frequent pair
        max_count = max(pair_counts.values())
        most_frequent_pairs = [
            pair for pair in pair_counts if pair_counts[pair] == max_count
        ]

        # If multiple pairs have same frequency, return the one that appears first
        return min(most_frequent_pairs, key=lambda x: first_occurrences[x])

    def replace_pair(self, text, pair, new_symbol):
        """
        Replace all occurrences of pair with new_symbol in text.
        Non-overlapping replacements from left to right.
        """
        result = []
        i = 0
        while i < len(text):
            if i < len(text) - 1 and text[i : i + 2] == pair:
                result.append(str(new_symbol))
                i += 2
            else:
                result.append(text[i])
                i += 1
        return "".join(result)

    def perform_bpe(self, text, k):
        """
        Perform k iterations of byte pair encoding on the input text.
        Returns both the encoded text and the mappings dictionary.
        """
        self.mappings = {}
        self.next_symbol = 0
        current_text = text

        # Perform k iterations or until no more pairs can be merged
        for _ in range(k):
            most_freq_pair = self.find_most_frequent_pair(current_text)

            # If no pairs found, break
            if most_freq_pair is None:
                break

            # Create new mapping
            new_symbol = str(self.next_symbol)
            self.mappings[new_symbol] = most_freq_pair
            self.next_symbol += 1

            # Replace pairs in text
            current_text = self.replace_pair(current_text, most_freq_pair, new_symbol)

        return {"encoded": current_text, "mappings": self.mappings}
