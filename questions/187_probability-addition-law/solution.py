def prob_union(p_a: float, p_b: float, p_intersection: float) -> float:
	"""Reference implementation for P(A ∪ B) with auto-detection of mutual exclusivity.

	If p_intersection is numerically very small (≤ 1e-12), treat as 0 and
	use the simplified rule P(A ∪ B) = P(A) + P(B).
	"""
	epsilon = 1e-12
	if p_intersection <= epsilon:
		return p_a + p_b
	return p_a + p_b - p_intersection
