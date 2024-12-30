def model_fit_quality(training_accuracy: float, test_accuracy: float) -> int:
    """
    Determine if the model is overfitting, underfitting, or a good fit based on training and test accuracy.
    :param training_accuracy: float, training accuracy of the model (0 <= training_accuracy <= 1)
    :param test_accuracy: float, test accuracy of the model (0 <= test_accuracy <= 1)
    :return: int, one of '1' (overfitting), '-1' (underfitting), or '0' (good fit).
    """
    if training_accuracy - test_accuracy > 0.2:
        return 1
    elif training_accuracy < 0.7 and test_accuracy < 0.7:
        return -1
    else:
        return 0

def test_model_fit_quality() -> None:
    # Test case 1: Overfitting
    result1 = model_fit_quality(0.9, 0.6)
    assert result1 == 1, f"Test case 1 failed: {result1} != 1"

    # Test case 2: Underfitting
    result2 = model_fit_quality(0.6, 0.5)
    assert result2 == -1, f"Test case 2 failed: {result2} != -1"

    # Test case 3: Good fit
    result3 = model_fit_quality(0.8, 0.75)
    assert result3 == 0, f"Test case 3 failed: {result3} != 0"

    # Test case 4: Borderline good fit
    result4 = model_fit_quality(0.7, 0.7)
    assert result4 == 0, f"Test case 4 failed: {result4} != 0"

    # Test case 5: Borderline overfitting
    result5 = model_fit_quality(0.9, 0.7)
    assert result5 == 0, f"Test case 5 failed: {result5} != 0"

if __name__ == "__main__":
    test_model_fit_quality()
    print("All tests passed.")
