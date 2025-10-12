## Solution Explanation

The probability addition law for any two events A and B states:

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

- The union counts outcomes in A or B (or both).
- We subtract the intersection once to correct double-counting.

### Mutually exclusive (disjoint) events
If A and B cannot occur together, then \(P(A \cap B) = 0\) and the addition rule simplifies to:
\[
P(A \cup B) = P(A) + P(B)
\]

### Plug in the given values

Given: \(P(A)=0.6\), \(P(B)=0.5\), \(P(A \cap B)=0.3\)

\[
P(A \cup B) = 0.6 + 0.5 - 0.3 = 0.8
\]

### Validity checks
- Probabilities must lie in [0, 1]. The result 0.8 is valid.
- Given inputs must satisfy: \(0 \le P(A \cap B) \le \min\{P(A), P(B)\}\) and \(P(A \cap B) \ge P(A) + P(B) - 1\). Here, 0.3 is within [0.1, 0.5], so inputs are consistent.

### Implementation outline
- Accept three floats: `p_a`, `p_b`, `p_intersection`.
- Optionally assert basic bounds to help users catch mistakes.
- Return `p_a + p_b - p_intersection`.
