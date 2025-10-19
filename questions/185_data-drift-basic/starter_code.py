from typing import List, Tuple


def check_drift(ref: List[float], cur: List[float], mean_threshold: float, var_threshold: float) -> Tuple[bool, bool]:
	"""Return (mean_drift, var_drift) comparing ref vs cur with given thresholds.

	Use population variance.
	"""
	# TODO: handle empty inputs; compute means and variances; compare with thresholds
	raise NotImplementedError
