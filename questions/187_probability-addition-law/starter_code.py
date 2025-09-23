# Implement your function below.

def prob_union(p_a: float, p_b: float, p_intersection: float) -> float:
	"""Return P(A ∪ B) using the addition law.

	Auto-detects mutually exclusive events by treating very small P(A ∩ B) as 0.

	Arguments:
	- p_a: P(A)
	- p_b: P(B)
	- p_intersection: P(A ∩ B)

	Returns:
	- float: P(A ∪ B)
	"""
	# TODO: if p_intersection is ~0, return p_a + p_b; else return p_a + p_b - p_intersection
	raise NotImplementedError
