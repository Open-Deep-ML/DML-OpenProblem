from typing import List, Tuple


def _mean(xs: List[float]) -> float:
	return sum(xs) / len(xs) if xs else 0.0


def _var(xs: List[float]) -> float:
	if not xs:
		return 0.0
	m = _mean(xs)
	return sum((x - m) * (x - m) for x in xs) / len(xs)


def check_drift(ref: List[float], cur: List[float], mean_threshold: float, var_threshold: float) -> Tuple[bool, bool]:
	if not ref or not cur:
		return (False, False)
	mean_ref = _mean(ref)
	mean_cur = _mean(cur)
	var_ref = _var(ref)
	var_cur = _var(cur)
	mean_drift = abs(mean_ref - mean_cur) > mean_threshold
	var_drift = abs(var_ref - var_cur) > var_threshold
	return (mean_drift, var_drift)
