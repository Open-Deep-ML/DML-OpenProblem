## Task: Generate a Composite Hypervector Using Hyperdimensional Computing

Your task is to implement the function `create_row_hv(row, dim, random_seeds)` to generate a composite hypervector for a given dataset row using Hyperdimensional Computing (HDC). Each feature in the row is represented by binding hypervectors for the feature name and its value. The hypervectors for the values are created using the same feature seed provided in the `random_seeds` dictionary to ensure reproducibility. All feature hypervectors are then bundled to create a composite hypervector for the row.

### Input:
- `row`: A dictionary representing a dataset row, where keys are feature names and values are their corresponding values.
- `dim`: The dimensionality of the hypervectors.
- `random_seeds`: A dictionary where keys are feature names and values are seeds to ensure reproducibility of hypervectors.

### Output:
- A composite hypervector representing the entire row.
