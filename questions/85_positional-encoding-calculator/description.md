Write a Python function to implement the Positional Encoding layer for Transformers.
      The function should calculate positional encodings for a sequence length (`position`) and model dimensionality (`d_model`) using sine and cosine functions as specified in the Transformer architecture.
      The function should return -1 if `position` is 0, or if `d_model` is less than or equal to 0. The output should be a numpy array of type `float16`.
