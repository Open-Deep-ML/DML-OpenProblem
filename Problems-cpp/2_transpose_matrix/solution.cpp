#include <iostream>
#include <vector>
#include <stdexcept>
#include <cassert>

template <typename T>
std::vector<std::vector<T>> transpose_matrix(const std::vector<std::vector<T>>& matrix);

template <typename T>
std::vector<std::vector<T>> transpose_matrix(const std::vector<std::vector<T>>& matrix)
{
    // Checking for an empty matrix or an empty row
    if (matrix.empty() || matrix[0].empty())  
    {
        throw std::invalid_argument("The matrix is empty");
    }
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<std::vector<T>> result(cols, std::vector<T>(rows, 0));
    for (size_t i = 0; i < rows; i++) 
    {
        for (size_t j = 0; j < cols; j++) 
        {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

void test_transpose_matrix()
{
    // Test case: Empty matrix
    try
    {
        std::cout << "Empty matrix: ";
        auto result = transpose_matrix<int>({});
        for (const auto& row : result)
        {
            for (const auto& val : row)
            {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }
    catch (const std::invalid_argument&)
    {
        std::cout << "Invalid matrix!" << std::endl;
    }

    // Test case: Single empty row matrix
    try
    {
        std::cout << "Matrix with one empty row: ";
        auto result = transpose_matrix<int>({{}});  // 1x0 matrix
        for (const auto& row : result)
        {
            for (const auto& val : row)
            {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }
    catch (const std::invalid_argument&)
    {
        std::cout << "Invalid matrix!" << std::endl;
    }

    // Valid product test case
    try
    {
        std::cout << "Valid product: ";
        std::vector<std::vector<int>> mat = {{1, 2, 3}, {4, 5, 6}};
        auto result = transpose_matrix<int>(mat);
        for (const auto& row : result)
        {
            for (const auto& val : row)
            {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<int>>({{1, 4}, {2, 5}, {3, 6}}));  
    }
    catch (const std::invalid_argument&)
    {
        std::cout << "Wrong product!" << std::endl;
    }
}

int main()
{
    test_transpose_matrix();
    return 0;
}
