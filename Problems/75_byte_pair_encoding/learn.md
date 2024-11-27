# Byte Pair Encoding (BPE) for Subword Tokenization

## Introduction
Byte Pair Encoding (BPE) is a subword-based tokenization technique that addresses the challenges posed by purely word-based or character-based tokenization methods. It was initially designed as a data compression algorithm but is now widely used in Natural Language Processing (NLP) to create compact and meaningful token vocabularies.

---

## Key Features of BPE
- **Balances between word and character tokenization:**
  - Word-based tokenization struggles with out-of-vocabulary (OOV) words and requires a large vocabulary.
  - Character-based tokenization can lead to long sequences and loss of semantic meaning in individual tokens.
- **Compact Vocabulary:**
  Merges frequently occurring subword units to reduce token count while preserving meaningful representations.
- **Dynamic Vocabulary Creation:**
  Iteratively merges the most common adjacent subword pairs to create a compressed vocabulary tailored to the corpus.

---

## Algorithm Overview

### 1. Input Preparation
- Add a special end-of-word token (`</w>`) to denote word boundaries in the corpus.  
  **Example:**  
  Words `old, older, finest, lowest` become:  
  `old</w>, older</w>, finest</w>, lowest</w>`

- Split each word into characters to initialize the token vocabulary.  
  **Initial tokens:** `{o, l, d, </w>, e, r, f, i, n, s, t}`

---

### 2. Iterative Merging Process
- **Step 1: Count Character Pair Frequencies**  
  Identify and count adjacent character pairs in the corpus.

- **Step 2: Merge the Most Frequent Pair**  
  Replace all occurrences of the most frequent pair with a new token. Add the merged token to the vocabulary.

- **Step 3: Update Frequencies**  
  Recompute character pair frequencies for the updated corpus.

- **Step 4: Repeat Until Stopping Criterion**  
  Continue merging until:
  - A fixed number of merges is reached.
  - A target vocabulary size is achieved.

---

## Detailed Example

### Corpus
`{"old</w>": 7, "older</w>": 3, "finest</w>": 9, "lowest</w>": 4}`

1. **Initialize Tokens and Frequencies**  
   Tokens: `{o, l, d, </w>, e, r, f, i, n, s, t}`

2. **Iterations**
   - **Iteration 1:**
     - Most frequent pair: `e` and `s` (frequency = 13).  
     - Merge to create token: `es`.  
     - Update corpus:  
       `{"old</w>": 7, "older</w>": 3, "finest</w>": 9 -> f i n es t, "lowest</w>": 4 -> l o w es t}`

   - **Iteration 2:**
     - Most frequent pair: `es` and `t` (frequency = 13).  
     - Merge to create token: `est`.  
     - Update corpus:  
       `{"old</w>": 7, "older</w>": 3, "finest</w>": 9 -> f i n est, "lowest</w>": 4 -> l o w est}`

   - **Iteration 3:**
     - Most frequent pair: `est` and `</w>` (frequency = 13).  
     - Merge to create token: `est</w>`.  
     - Update corpus:  
       `{"old</w>": 7, "older</w>": 3, "finest</w>": 9 -> f i n est</w>, "lowest</w>": 4 -> l o w est</w>}`

   - **Iteration 4:**
     - Most frequent pair: `o` and `l` (frequency = 10).  
     - Merge to create token: `ol`.  
     - Update corpus:  
       `{"old</w>": 7 -> ol d</w>, "older</w>": 3 -> ol d e r</w}, ...`

   - **Iteration 5:**
     - Most frequent pair: `ol` and `d` (frequency = 10).  
     - Merge to create token: `old`.  
     - Update corpus:  
       `{"old</w>": 7 -> old</w>, "older</w>": 3 -> old e r</w}, ...`

---

## Final Vocabulary
After completing the iterations, we obtain a vocabulary of subword units such as:  
`{o, l, d, </w>, e, r, f, i, n, s, t, es, est, est</w>, ol, old}`

This vocabulary efficiently represents the original corpus while being compact and meaningful.

---

## Advantages of BPE
1. **Handles OOV Words:**  
   Rare or unseen words are represented as a combination of subwords, avoiding the need for large vocabularies.
2. **Compact Representation:**  
   Reduces the number of tokens required while maintaining semantic fidelity.
3. **Adaptability:**  
   Can be applied to multiple languages and custom datasets.

---

## Applications of BPE
- **NLP Models:** Tokenization for models like GPT and BERT.
- **Machine Translation:** Subword units provide better alignment and understanding.
- **Data Compression:** Originally designed for data compression tasks.

