def dot(v1:list[float], v2:list[float]) -> float:

    if len(v1) != len(v2):
        raise Exception("Vectors are not of the same dimensions.")
    
    return sum([ax1*ax2 for ax1,ax2 in zip(v1,v2)])

def scalar_mult(scalar:float, v:list[float]) -> list[float]:


    return [scalar*ax for ax in v]

def orthogonal_projection(v:list[float], L: list[float]) -> list[list[float]]:
    
    # calculate the orthogonal projection of v onto L
    # calculate the unit vector of L
    L_mag = sum([ax**2 for ax in L])**0.5
    u = [ax/L_mag for ax in L]

    # calculate orthogonal projection
    proj_v = scalar_mult(dot(v, u), u)

    return proj_v

def test_orthogonal_projection() -> None:
    # Test case 1: 2D vectors
    v1 = [3, 4]
    L1 = [1, 0]
    expected_proj1 = [3, 0]  # Projection of v1 onto L1 should lie along the x-axis
    assert orthogonal_projection(v1, L1) == expected_proj1, f"Test case 1 failed: {orthogonal_projection(v1, L1)} != {expected_proj1}"

    # Test case 2: 3D vectors
    v2 = [1, 2, 3]
    L2 = [0, 0, 1]
    expected_proj2 = [0, 0, 3]  # Projection of v2 onto L2 should be along the z-axis
    assert orthogonal_projection(v2, L2) == expected_proj2, f"Test case 2 failed: {orthogonal_projection(v2, L2)} != {expected_proj2}"

    # Test case 3: Arbitrary 3D vectors
    v3 = [5, 6, 7]
    L3 = [2, 0, 0]  # Projection should align with the x-axis
    expected_proj3 = [5, 0, 0]
    assert orthogonal_projection(v3, L3) == expected_proj3, f"Test case 3 failed: {orthogonal_projection(v3, L3)} != {expected_proj3}"

if __name__ == "__main__":
    test_orthogonal_projection()
    print("All tests passed.")
