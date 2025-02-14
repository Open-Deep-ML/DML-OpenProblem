# Byte Pair Encoding Solution Explanation

## Overview
The solution implements Byte Pair Encoding (BPE) through a class-based approach with three main components:
1. Finding the most frequent pair
2. Replacing pairs with new symbols
3. Iterative encoding process

## Detailed Component Breakdown

### 1. BytePairEncoder Class
```python
class BytePairEncoder:
    def __init__(self):
        self.mappings = {}
        self.next_symbol = 0
```
- `mappings`: Dictionary storing the replacement history
- `next_symbol`: Counter for generating new symbols (0, 1, 2, etc.)

### 2. Finding Most Frequent Pair
```python
def find_most_frequent_pair(self, text):
```

This method works in three steps:
1. **Counting Frequencies**:
   ```python
   pair_counts = {}
   first_occurrences = {}
   for i in range(len(text) - 1):
       pair = text[i:i+2]
       if pair not in first_occurrences:
           first_occurrences[pair] = i
       pair_counts[pair] = pair_counts.get(pair, 0) + 1
   ```
   - Tracks both frequency and first occurrence position
   - First occurrence is used for tie-breaking

2. **Finding Maximum Frequency**:
   ```python
   max_count = max(pair_counts.values())
   most_frequent_pairs = [pair for pair in pair_counts if pair_counts[pair] == max_count]
   ```
   - Creates list of all pairs with highest frequency

3. **Tie Breaking**:
   ```python
   return min(most_frequent_pairs, key=lambda x: first_occurrences[x])
   ```
   - Returns pair that appears first in original text

### 3. Replacing Pairs
```python
def replace_pair(self, text, pair, new_symbol):
```

The replacement process:
1. **Non-overlapping Iteration**:
   ```python
   while i < len(text):
       if i < len(text) - 1 and text[i:i+2] == pair:
           result.append(str(new_symbol))
           i += 2
       else:
           result.append(text[i])
           i += 1
   ```
   - Processes text left to right
   - Ensures non-overlapping replacements
   - Handles single characters properly

### 4. Main BPE Process
```python
def perform_bpe(self, text, k):
```

Iterative process:
1. **Initialization**:
   - Resets mappings and symbol counter
   - Sets up initial text

2. **Iteration Loop**:
   ```python
   for _ in range(k):
       most_freq_pair = self.find_most_frequent_pair(current_text)
       if most_freq_pair is None:
           break
   ```
   - Performs up to k iterations
   - Stops if no pairs are found

3. **Symbol Assignment**:
   ```python
   new_symbol = str(self.next_symbol)
   self.mappings[new_symbol] = most_freq_pair
   self.next_symbol += 1
   ```
   - Assigns sequential numbers as new symbols
   - Records mapping for reference

## Time Complexity Analysis
- Finding Most Frequent Pair: O(n) where n is text length
- Replacing Pairs: O(n)
- Overall Process: O(k * n) where k is number of iterations

## Space Complexity Analysis
- Mappings Dictionary: O(k) where k is number of iterations
- Pair Counts: O(n) where n is text length
- Total Space: O(n + k)

## Example Walkthrough
Let's walk through test case 1: `"aabaabaab"` with `k=2`

1. **First Iteration**:
   - Counts: `{"aa": 3, "ab": 3, "ba": 2}`
   - First occurrences: `{"aa": 0, "ab": 1, "ba": 2}`
   - `"aa"` is chosen (appears first)
   - Text becomes: `"0b0b0b"` (0 represents "aa")

2. **Second Iteration**:
   - Counts: `{"0b": 3}`
   - Replace `"0b"` with `"2"`
   - Final text: `"222"`

3. **Final Output**:
   ```python
   {
       "encoded": "222",
       "mappings": {
           "0": "aa",
           "1": "ab",
           "2": "0b"
       }
   }
   ```

## Common Edge Cases Handled
1. No repeating pairs
2. Single character text
3. Multiple pairs with same frequency
4. Overlapping pairs
5. k larger than possible merges