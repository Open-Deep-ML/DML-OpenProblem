"""
Generate test cases for the Mahalanobis distance problem.
This script creates comprehensive test cases and writes them to tests.json.
"""
import json
import numpy as np
from pathlib import Path


def mahalanobis_distance_reference(point, mean, cov_matrix):
    """Reference implementation for generating expected outputs."""
    diff = point - mean
    try:
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_matrix)
    distance = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
    return float(distance)


def generate_tests():
    """Generate diverse test cases for Mahalanobis distance."""
    tests = []
    
    # Test 1: Identity covariance (Euclidean distance)
    point1 = np.array([3.0, 4.0])
    mean1 = np.array([0.0, 0.0])
    cov1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    expected1 = round(mahalanobis_distance_reference(point1, mean1, cov1), 4)
    tests.append({
        "test": "print(mahalanobis_distance(np.array([3.0, 4.0]), np.array([0.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0]])))",
        "expected_output": str(expected1)
    })
    
    # Test 2: Scaled identity covariance (2D)
    point2 = np.array([2.0, 2.0])
    mean2 = np.array([0.0, 0.0])
    cov2 = np.array([[4.0, 0.0], [0.0, 1.0]])
    expected2 = round(mahalanobis_distance_reference(point2, mean2, cov2), 4)
    tests.append({
        "test": "print(mahalanobis_distance(np.array([2.0, 2.0]), np.array([0.0, 0.0]), np.array([[4.0, 0.0], [0.0, 1.0]])))",
        "expected_output": str(expected2)
    })
    
    # Test 3: Non-diagonal covariance (with correlation)
    point3 = np.array([1.0, 1.0])
    mean3 = np.array([0.0, 0.0])
    cov3 = np.array([[2.0, 1.0], [1.0, 2.0]])
    expected3 = round(mahalanobis_distance_reference(point3, mean3, cov3), 4)
    tests.append({
        "test": "print(mahalanobis_distance(np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([[2.0, 1.0], [1.0, 2.0]])))",
        "expected_output": str(expected3)
    })
    
    # Test 4: 3D case with non-zero mean
    point4 = np.array([5.0, 3.0, 2.0])
    mean4 = np.array([1.0, 1.0, 1.0])
    cov4 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    expected4 = round(mahalanobis_distance_reference(point4, mean4, cov4), 4)
    tests.append({
        "test": "print(mahalanobis_distance(np.array([5.0, 3.0, 2.0]), np.array([1.0, 1.0, 1.0]), np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])))",
        "expected_output": str(expected4)
    })
    
    # Test 5: 3D with scaled covariance
    point5 = np.array([2.0, 2.0, 2.0])
    mean5 = np.array([0.0, 0.0, 0.0])
    cov5 = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    expected5 = round(mahalanobis_distance_reference(point5, mean5, cov5), 4)
    tests.append({
        "test": "print(mahalanobis_distance(np.array([2.0, 2.0, 2.0]), np.array([0.0, 0.0, 0.0]), np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])))",
        "expected_output": str(expected5)
    })
    
    # Test 6: Point at mean (distance should be 0)
    point6 = np.array([1.0, 2.0])
    mean6 = np.array([1.0, 2.0])
    cov6 = np.array([[1.0, 0.0], [0.0, 1.0]])
    expected6 = round(mahalanobis_distance_reference(point6, mean6, cov6), 4)
    tests.append({
        "test": "print(mahalanobis_distance(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([[1.0, 0.0], [0.0, 1.0]])))",
        "expected_output": str(expected6)
    })
    
    # Test 7: 4D case
    point7 = np.array([1.0, 2.0, 3.0, 4.0])
    mean7 = np.array([0.0, 0.0, 0.0, 0.0])
    cov7 = np.eye(4)
    expected7 = round(mahalanobis_distance_reference(point7, mean7, cov7), 4)
    tests.append({
        "test": "print(mahalanobis_distance(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 0.0, 0.0, 0.0]), np.eye(4)))",
        "expected_output": str(expected7)
    })
    
    # Test 8: Negative coordinates
    point8 = np.array([-2.0, 3.0])
    mean8 = np.array([1.0, 1.0])
    cov8 = np.array([[1.0, 0.5], [0.5, 1.0]])
    expected8 = round(mahalanobis_distance_reference(point8, mean8, cov8), 4)
    tests.append({
        "test": "print(mahalanobis_distance(np.array([-2.0, 3.0]), np.array([1.0, 1.0]), np.array([[1.0, 0.5], [0.5, 1.0]])))",
        "expected_output": str(expected8)
    })
    
    return tests


def main():
    """Generate tests and write to tests.json."""
    tests = generate_tests()
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    output_file = script_dir / "tests.json"
    
    # Write to JSON file
    with open(output_file, "w") as f:
        json.dump(tests, f, indent=2)
    
    print(f"✓ Generated {len(tests)} test cases")
    print(f"✓ Written to {output_file}")


if __name__ == "__main__":
    main()
