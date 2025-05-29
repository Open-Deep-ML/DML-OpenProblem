#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cassert>

template <typename T>
std::vector <double> calculate_matrix_mean(const std::vector <std::vector <T>> &matrix, const std::string &str)
{
    if (str != "column" && str != "row")
    {
        throw std::invalid_argument("The string should either be \"column\" or \"row\".");
    }
    if (matrix.empty())
    {
        throw std::invalid_argument("The matrix is empty!");
    }
    int rows = matrix.size();
    int cols = matrix[0].size();
    if (str == "column")
    {
        std::vector <double> mean_vector(cols, 0);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                mean_vector[j] += matrix[i][j];
            }
        }
        for (int k = 0; k < cols; k++)
        {
            mean_vector[k]/= rows;
        }
        return mean_vector;
    }
    else if (str == "row")
    {
        std::vector <double> mean_vector(rows, 0);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
               mean_vector[i] += matrix[i][j];
            }
            mean_vector[i]/=cols;
        }
        return mean_vector;
    }
    return std::vector <double> {};
}

void test_calculate_matrix_mean()
{
    try
    {
        std::cout << "Case 1" << std::endl;
        std::vector <std::vector <int>> mat = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        std::string type = "row";
        auto result = calculate_matrix_mean(mat, type);
        for (auto val:result) std::cout << val << " ";
        std::cout << std::endl;
        assert(result == std::vector <double> ({2, 5, 8}));
    }
    catch(const std::invalid_argument &)
    {
        std::cout << "The mean vector is not correct" << '\n';
    }
    try
    {
        std::cout << "Case 2" << std::endl;
        std::vector <std::vector <int>> mat = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        std::string type = "column";
        auto result = calculate_matrix_mean(mat, type);
        for (auto val:result) std::cout << val << " ";
        std::cout << std::endl;
        assert(result == std::vector <double> ({4, 5, 6}));
    }
    catch(const std::invalid_argument &)
    {
        std::cout << "The mean vector is not correct" << '\n';
    }
    try
    {
        std::cout << "Case 3" << std::endl;
        std::vector <std::vector <int>> mat = {{1, 2}, {3, 4}, {5, 6}};
        std::string type = "column";
        auto result = calculate_matrix_mean(mat, type);
        for (auto val:result) std::cout << val << " ";
        std::cout << std::endl;
        assert(result == std::vector <double> ({3, 4}));
    }
    catch(const std::invalid_argument &)
    {
        std::cout << "The mean vector is not correct" << '\n';
    }
    try
    {
        std::cout << "Case 4" << std::endl;
        std::vector <std::vector <int>> mat = {{1, 2}, {3, 4}, {5, 6}};
        std::string type = "row";
        auto result = calculate_matrix_mean(mat, type);
        for (auto val:result) std::cout << val << " ";
        std::cout << std::endl;
        assert(result == std::vector <double> ({1.5, 3.5, 5.5}));
    }
    catch(const std::invalid_argument &)
    {
        std::cout << "The mean vector is not correct" << '\n';
    }
    try
    {
        std::cout << "Case 5" << std::endl;
        std::vector <std::vector <int>> mat = {{1}, {3}};
        std::string type = "rowl";
        auto result = calculate_matrix_mean(mat, type);
        for (auto val:result) std::cout << val << " ";
        std::cout << std::endl;
        //assert(result == std::vector <double> ({1.5, 3.5, 5.5}));
    }
    catch(const std::invalid_argument &)
    {
        std::cout << "It should be either \"row\" or \"column\"!" << '\n';
    }
}

int main()
{
    test_calculate_matrix_mean();
    return 0;
}
