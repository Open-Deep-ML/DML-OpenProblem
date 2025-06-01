#include <iostream>
#include <stdexcept>
#include <cassert>
#include <type_traits>
#include <cmath>

template <typename T, typename U>
std::vector <std::vector <typename std::common_type <T, U>::type>> scalar_multiply(std::vector <std::vector <T>> &matrix, U scalar)
{
    using resultType = typename std::common_type<T, U>::type;
    if (matrix.empty())
    {
        throw std::invalid_argument("The matrix is empty!");
    }
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector <std::vector <resultType>> result(rows, std::vector <resultType>(cols, 1));
    for (int i = 0; i < rows; i++)
    {
        for(int j = 0; j <cols; j++)
        {
            result[i][j] = scalar * matrix[i][j];   
        }
    }
    return result;
}

void test_scalar_multiply()
{
    try
    {
        std::cout << "Case 1:" << std::endl;
        std::vector <std::vector <int>> mat = {{1, 2}, {3, 4}};
        auto result = scalar_multiply(mat, 2);
        for (auto row: result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        assert(result == std::vector <std::vector <int>>({{2,4},{6, 8}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cout << "The answer is not correct!" << std::endl;
    }
    try
    {
        std::cout << "Case 2:" << std::endl;
        std::vector <std::vector <int>> mat = {{0, -1}, {1, 0}};
        auto result = scalar_multiply(mat, -1);
        for (auto row: result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        assert(result == std::vector <std::vector <int>>({{0, 1},{-1, 0}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cout << "The answer is not correct!" << std::endl;
    }
    try
    {
        std::cout << "Case 3:" << std::endl;
        std::vector <std::vector <int>> mat = {{}};
        auto result = scalar_multiply(mat, -1);
        for (auto row: result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
        //assert(result == std::vector <std::vector <int>>({{0, 1},{-1, 0}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cout << "The matrix is actually empty!" << std::endl;
    }
    try
    {
        std::cout << "Case 4:" << std::endl;
        std::vector <std::vector <double>> mat = {{2.1, 3.2, 3.1}, {1.4, 2.5, -1.5}};
        auto result = scalar_multiply(mat, -0.75);
        for (auto row: result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(std::abs(result[0][0] - (-1.575)) < 1e-9 &&
               std::abs(result[0][1] - (-2.4)) < 1e-9 &&
               std::abs(result[1][2] - (1.125)) < 1e-9);
    }
    catch(const std::invalid_argument &)
    {
        std::cout << "The answer is not correct!" << std::endl;
    }
}

int main()
{
    test_scalar_multiply();
    return 0;
}