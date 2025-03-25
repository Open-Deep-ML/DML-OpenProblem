#include <iostream>
#include <vector>
#include <cmath> 
#include <cassert> 

using namespace std;

vector<double> calculate_eigenvalues(const vector<vector<double> >& matrix)
{
    // Extract elements
    double a = matrix[0][0];
    double b = matrix[0][1];
    double c = matrix[1][0];
    double d = matrix[1][1];

    // Get trace and determinant
    double trace = a + d;
    double determinant = a * d - b * c;
    double discriminant = trace * trace - 4.0 * determinant;

    // Solve for eigenvalues
    double lambda1 = (trace + sqrt(discriminant)) / 2.0;
    double lambda2 = (trace - sqrt(discriminant)) / 2.0;

    // Return as vector
    vector<double> eigenvalues(2);
    eigenvalues[0] = lambda1;
    eigenvalues[1] = lambda2;
    return eigenvalues;
}


void test_calculate_eigenvalues()
{
    // Test Case 1
    {

        vector<vector<double> > matrix(2, vector<double>(2));
        matrix[0][0] = 2.0;  matrix[0][1] = 1.0;
        matrix[1][0] = 1.0;  matrix[1][1] = 2.0;

        vector<double> result = calculate_eigenvalues(matrix);
        assert(result.size() == 2);

        assert(result[0] == 3.0);
        assert(result[1] == 1.0);
    }

    // Test Case 2
    {

        vector<vector<double> > matrix(2, vector<double>(2));
        matrix[0][0] = 4.0;   matrix[0][1] = -2.0;
        matrix[1][0] = 1.0;   matrix[1][1] = 1.0;

        vector<double> result = calculate_eigenvalues(matrix);
        assert(result.size() == 2);

        assert(result[0] == 3.0);
        assert(result[1] == 2.0);
    }
}

int main()
{
    test_calculate_eigenvalues();
    return 0;
}
