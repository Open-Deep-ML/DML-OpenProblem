## Problem

Two events `A` and `B` in a probability space have the following probabilities:

- P(A) = 0.6
- P(B) = 0.5
- P(A ∩ B) = 0.3

Using the probability addition law, compute `P(A ∪ B)`.

Implement a function `prob_union(p_a, p_b, p_intersection)` that returns `P(A ∪ B)` as a float.

Recall: P(A ∪ B) = P(A) + P(B) − P(A ∩ B).

Note: If `A` and `B` are mutually exclusive (disjoint), then `P(A ∩ B) = 0` and the rule simplifies to `P(A ∪ B) = P(A) + P(B)`.
