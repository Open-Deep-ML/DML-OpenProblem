#include <iostream>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <cassert>
template <typename T, typename U>
std::vector<typename std::common_type<T, U>::type> matrix_dot_vector(const std::vector <std::vector <T>> &, const std::vector <U> &);
void test_matrix_vector_product(); 


template <typename T, typename U>
std::vector<typename std::common_type<T, U>::type> matrix_dot_vector(const std::vector <std::vector <T>> &a, const std::vector <U> &b)
{
    // Get the common type of T, U, if T:<double>, and U:<int>, common type would be double. 
    using ResultType = typename std::common_type<T, U>::type;
    // Check if the matrix and the vector are actually empty!
    if (a.empty()||b.empty())
    {
        throw std::invalid_argument("The matrix or the vector are empty!");
    }
    // Check if the number of cols of the matrix are equal to the number of rows in the vector
    if (a[0].size() != b.size())
    {
        throw std::invalid_argument("The dimensions of the matrix and the vector do not match!");
    }
    // Create an empty vector of the common type initialised with 0.
    std::vector<ResultType> c(a.size(), 0);
    // Perform the Matrix-vector multiplication.
    for (size_t i = 0; i < a.size(); i++)
    {
        for (size_t j = 0; j < a[i].size(); j++)
        {
            c[i] += a[i][j]*b[j];
        }
    }
    // return the result
    return c;
}

void test_matrix_vector_product()
{
   // Empty product (invalid case)
    try {
        std::cout << "Empty product: ";
        auto result = matrix_dot_vector<int, int>({}, {});  // This creates empty matrix and vector
        for (auto val : result) std::cout << val << " ";
        std::cout << std::endl;
    } catch (const std::invalid_argument&) {
        std::cout << "Invalid product" << std::endl;
    }
    
    // Invalid product (dimension mismatch)
    try {
        std::cout << "Invalid product: ";
        auto result = matrix_dot_vector<int, int>({{1, 2}}, {});
        for (auto val : result) std::cout << val << " ";
        std::cout << std::endl;
    } catch (const std::invalid_argument&) {
        std::cout << "Invalid product" << std::endl;
    }
    
    // Valid product
    try{
        std::vector<std::vector<int>> a = {{1, 2}, {2, 4}};
        std::vector<int> b = {1, 2};
        auto result = matrix_dot_vector(a, b);
        std::cout << "Valid product: ";
        for (auto val : result) std::cout << val << " ";
        std::cout << std::endl;
        assert(result == std::vector <int> ({5, 10}));
    } catch (const std::invalid_argument&)
    {
        std::cout << "Wrong product!" << std::endl;
    }

    // Valid product with rectangular matrix
    try
    {
        std::vector<std::vector<int>> a2 = {{1, 2, 3}, {2, 4, 6}};
        std::vector<int> b2 = {1, 2, 3};
        auto result2 = matrix_dot_vector(a2, b2);
        std::cout << "Valid product (rectangular matrix): ";
        for (auto val : result2) std::cout << val << " ";
        std::cout << std::endl;
        assert(result2 == std::vector <int>({14, 28}));
    }
    catch (const std::invalid_argument&)
    {
        std::cout << "Wrong product!" << std::endl;
    }
}

int main()
{
    test_matrix_vector_product();
    return 0;
}