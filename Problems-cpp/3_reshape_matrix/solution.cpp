#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cassert>

template <typename T>

std::vector <std::vector <T>> reshape_matrix(std::vector <std::vector <T>> &matrix, int newrows, int newcols)
{
    // check if the matrix is empty
    if (matrix.empty())
    {
        throw std::invalid_argument("The matrix is empty! It cannot be reshaped.");
    }
    // get the number of rows and columns
    int rows = matrix.size();
    int cols = matrix[0].size();
    // check if it can be reshaped
    if (rows*cols != newrows*newcols)
    {
        throw std::invalid_argument("The matrix cannot be reshaped from (" + std::to_string(rows) + ", " 
        + std::to_string(cols) + ") to (" + std::to_string(newrows) +", " + std::to_string(newcols) + ")");
    }
    // check if the new shape is the same as the old shape
    if (rows == newrows && cols == newcols)
    {
        return matrix;
    }
    // Reshape the matrix
    std::vector <std::vector <T>> result(newrows, std::vector <T> (newcols, 0));
    for (int i = 0; i < newrows*newcols; i++)
    {
        int oldrow = i/cols;
        int oldcol = i%cols;
        int newrow = i/newcols;
        int newcol = i%newcols;
        result[newrow][newcol] = matrix[oldrow][oldcol];
    }
    return result;
}

void test_reshape_matrix()
{
    try
    {
        std::cout << "Case 1:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{1,2,3,4}, {5,6,7,8}};
        auto result = reshape_matrix(mat1, 4, 2);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<int>>({{1,2},{3,4},{5,6},{7,8}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshaped the wrong way!" << std::endl;
    }
    try
    {
        std::cout << "Case 2:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{1,2,3}, {4,5,6}};
        auto result = reshape_matrix(mat1, 3, 2);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<int>>({{1,2},{3,4},{5,6}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshaped the wrong way!" << std::endl;
    }
    try
    {
        std::cout << "Case 3:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{1,2,3,4}, {5,6,7,8}};
        auto result = reshape_matrix(mat1, 2, 4);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<int>>({{1,2,3,4},{5,6,7,8}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshaped the wrong way!" << std::endl;
    }
    try
    {
        std::cout << "Case 4:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{1,2,3,4}, {5,6,7,8}};
        auto result = reshape_matrix(mat1, 1, 4);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Cannot be reshaped!" << std::endl;
    }
    try
    {
        std::cout << "Case 5:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{1,2}, {3,4}};
        auto result = reshape_matrix(mat1, 4, 1);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<int>>({{1},{2}, {3}, {4}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshaped the wrong way!" << std::endl;
    }
    try
    {
        std::cout << "Case 6:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{1,2,3}, {4,5,6}};
        auto result = reshape_matrix(mat1, 2, 3);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<int>>({{1,2,3},{4,5,6}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshaped the wrong way!" << std::endl;
    }
    try
    {
        std::cout << "Case 7:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{1,2,3}, {4,5,6}};
        auto result = reshape_matrix(mat1, 6, 1);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<int>>({{1},{2}, {3}, {4}, {5},{6}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshaped the wrong way!" << std::endl;
    }
    try
    {
        std::cout << "Case 8:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{1,2}, {3,4}, {5,6}};
        auto result = reshape_matrix(mat1, 3, 2);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<int>>({{1,2},{3,4},{5,6}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshaped the wrong way!" << std::endl;
    }
    try
    {
        std::cout << "Case 9:" << std::endl;
        std::vector <std::vector <double>> mat1 = {{1.5, 2.2,3.1},{ 4.7, 5.9, 6.3}};
        auto result = reshape_matrix(mat1, 3, 2);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<double>>({{1.5,2.2},{3.1,4.7},{5.9, 6.3}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshaped the wrong way!" << std::endl;
    }
        try
    {
        std::cout << "Case 10:" << std::endl;
        std::vector <std::vector <double>> mat1 = {{1.5, 2.2, 3.1},{ 4.7, 5.9, 6.3}, {7.7, 8.8, 9.9}};
        auto result = reshape_matrix(mat1, 9, 1);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<double>>({{1.5},{2.2},{3.1},{4.7},{5.9}, {6.3}, {7.7}, {8.8},{9.9}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshaped the wrong way!" << std::endl;
    }
    try
    {
        std::cout << "Case 11:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{0, 0},{0, 0}};
        auto result = reshape_matrix(mat1, 2, 2);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<int>>({{0, 0},{0, 0}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshape has been done the wrong way!" << std::endl;
    }
    try
    {
        std::cout << "Case 12:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{1,2}};
        auto result = reshape_matrix(mat1, 2, 2);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        //assert(result == std::vector<std::vector<int>>({{0, 0},{0, 0},{0, 0}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Cannot be reshaped!" << std::endl;
    }
    try
    {
        std::cout << "Case 13:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{10, 20, 30, 40}};
        auto result = reshape_matrix(mat1, 2, 2);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<int>>({{10, 20},{30, 40}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshaped the wrong way!" << std::endl;
    }
    try
    {
        std::cout << "Case 14:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{1}, {2},{3}, {4}};
        auto result = reshape_matrix(mat1, 2, 2);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<int>>({{1, 2},{3, 4}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshaped the wrong way!" << std::endl;
    }
    try
    {
        std::cout << "Case 15:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{1},{2},{3}, {4}};
        auto result = reshape_matrix(mat1, 8, 1);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        //assert(result == std::vector<std::vector<int>>({{0, 0},{0, 0},{0, 0}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Cannot be reshaped!" << std::endl;
    }
    try
    {
        std::cout << "Case 16:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{1}, {2},{3}, {4}, {5}};
        auto result = reshape_matrix(mat1, 1, 5);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<int>>({{1,2,3,4,5}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshaped the wrong way!" << std::endl;
    }
    try
    {
        std::cout << "Case 17:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{-1, -2},{-3, -4}};
        auto result = reshape_matrix(mat1, 1, 4);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<int>>({{-1, -2, -3, -4}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshaped the wrong way!" << std::endl;
    }
    try
    {
        std::cout << "Case 18:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{-1, 2},{3, -4}, {5, 6}, {7, -8}};
        auto result = reshape_matrix(mat1, 2, 4);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<int>>({{-1, 2, 3, -4},{5, 6, 7, -8}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshaped the wrong way!" << std::endl;
    }
    try
    {
        std::cout << "Case 19:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{1}, {2}, {3}, {4}, {5}, {6}};
        auto result = reshape_matrix(mat1, 3, 2);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<int>>({{1, 2},{3, 4},{5, 6}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshaped the wrong way!" << std::endl;
    }
    try
    {
        std::cout << "Case 20:" << std::endl;
        std::vector <std::vector <int>> mat1 = {{1}, {2}, {3}, {4}, {5}, {6}};
        auto result = reshape_matrix(mat1, 2, 3);
        for (auto row:result)
        {
            for (auto col:row)
            {
                std::cout << col << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
        assert(result == std::vector<std::vector<int>>({{1, 2, 3},{ 4,5, 6}}));
    }
    catch(const std::invalid_argument &)
    {
        std::cerr << "Reshaped the wrong way!" << std::endl;
    }
}

int main()
{
    test_reshape_matrix();
    return 0;
}