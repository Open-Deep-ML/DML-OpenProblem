import numpy as np

def create_hv(dim):
    return np.random.choice([-1, 1], dim)

def create_col_hvs(dim, seed):
    np.random.seed(seed)
    return create_hv(dim), create_hv(dim)

def bind(hv1, hv2):
    return hv1*hv2

def bundle(hvs, dim):
    bundled = np.sum(list(hvs.values()), axis=0)
    return sign(bundled)

def sign(vector, threshold=0.01):
    return np.array([1 if v >= 0 else -1 for v in vector])

def create_row_hv(row:dict[str,str], dim:int, random_seeds:dict[str,int]):
    row_hvs = {col:bind(*create_col_hvs(dim, random_seeds[col])) for col in row.keys()}
    return bundle(row_hvs, dim)


def test_create_row_hv():
    # Define a sample row and a seed dictionary for reproducibility
    row = {"FeatureA": "value1", "FeatureB": "value2"}
    seed_dict = {"FeatureA": 42, "FeatureB": 7}

    # Test case 1
    dim = 5
    hv = create_row_hv(row, dim, seed_dict)
    result_hv = np.array([1,-1,1,1,1])
    assert len(hv) == dim, "Test case 1 failed: Incorrect dimension"
    assert np.all(np.isin(hv, [-1, 1])), "Test case 1 failed: Non-bipolar values present"
    assert  np.array_equal(hv,result_hv)

    # Test case 2
    dim = 10
    hv = create_row_hv(row, dim, seed_dict)
    result_hv = np.array([1,-1,1,1,-1,-1,-1,-1,-1,-1])
    assert len(hv) == dim, "Test case 2 failed: Incorrect dimension"
    assert np.all(np.isin(hv, [-1, 1])), "Test case 2 failed: Non-bipolar values present"
    assert  np.array_equal(hv,result_hv)

    # Test case 3
    dim = 15
    hv = create_row_hv(row, dim, seed_dict)
    result_hv = np.array([1,1,-1,-1,1,1,1,1,-1,1,1,1,-1,-1,1])
    assert len(hv) == dim, "Test case 3 failed: Incorrect dimension"
    assert np.all(np.isin(hv, [-1, 1])), "Test case 3 failed: Non-bipolar values present"
    assert  np.array_equal(hv,result_hv)

if __name__ == "__main__":
    test_create_row_hv()
    print("All recall tests passed.")
