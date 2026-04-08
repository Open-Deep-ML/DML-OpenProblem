import numpy as np

def mahalanobis_distance(point: np.ndarray, mean: np.ndarray, cov_matrix: np.ndarray) -> float:
	"""
	Calculate the Mahalanobis distance between a point and a probability distribution.
	
	Args:
		point (np.ndarray): A 1D numpy array representing the point in the feature space.
			Shape: (n_features,)
		mean (np.ndarray): A 1D numpy array representing the mean of the distribution.
			Shape: (n_features,)
		cov_matrix (np.ndarray): A 2D numpy array representing the covariance matrix.
			Shape: (n_features, n_features)
	
	Returns:
		float: The Mahalanobis distance, computed as sqrt((point - mean)^T * cov_matrix^(-1) * (point - mean))
			   A scalar representing how many standard deviations the point is from the mean,
			   accounting for correlation between features.
	
	"""
	# Compute deviation vector
	diff = point - mean
	
	# Compute inverse of covariance matrix
	try:
		inv_cov = np.linalg.inv(cov_matrix)
	except np.linalg.LinAlgError:
		# Handle singular covariance matrix using pseudo-inverse
		inv_cov = np.linalg.pinv(cov_matrix)
	
	# Compute Mahalanobis distance: sqrt((x - mu)^T * Sigma^(-1) * (x - mu))
	distance = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
	
	return round(float(distance), 4)
