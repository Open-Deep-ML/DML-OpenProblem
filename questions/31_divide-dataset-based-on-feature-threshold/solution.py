import numpy as np


def divide_on_feature(X, feature_i, threshold):
    # Define the split function based on the threshold type
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        # For numeric threshold, check if feature value is greater than or equal to the threshold
        def split_func(sample):
            return sample[feature_i] >= threshold
    else:
        # For non-numeric threshold, check if feature value is equal to the threshold
        def split_func(sample):
            return sample[feature_i] == threshold

    # Create two subsets based on the split function
    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    # Return the two subsets
    return [X_1, X_2]
