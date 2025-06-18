Create a function named sparse_window_attention that computes sparse attention over long sequences by sliding a fixed-radius window across the sequence.

• The parameter window_size represents the radius w of the window.
- For a token at index i, attend only to tokens whose indices are within max(0, i - w) through min(seq_len - 1, i + w), inclusive.
- Tokens near the beginning or end of the sequence simply have smaller windows; no padding is added.

• Inputs
- Q, K, V: NumPy arrays with shapes (seq_len, d_k) for Q and K, and (seq_len, d_v) for V.
- window_size: integer window radius.
- scale_factor (optional): value used to scale dot-product scores; if None, default to sqrt(d_k).

• Output
- A NumPy array of shape (seq_len, d_v) containing the attention results.
