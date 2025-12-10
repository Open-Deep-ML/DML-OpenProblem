#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

vector<vector<double> > matrixMul(const vector<vector<double> >& a, const vector<vector<double> >& b)
{
    if (a.size() == 0 || b.size() == 0 || a[0].size() != b.size())
    {
        vector<vector<double> > emptyResult;
        return emptyResult;
    }
    
    size_t rows = a.size();
    size_t cols = b[0].size();
    size_t inner = b.size();
    
    vector<vector<double> > result(rows, vector<double>(cols, 0));
    
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            for (size_t k = 0; k < inner; k++)
            {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    
    return result;
}

void testMatrixMul()
{
    // Test Case 1
    {
        vector<vector<double> > a(3, vector<double>(3));
        a[0] = vector<double>(3); a[0][0] = 1; a[0][1] = 2; a[0][2] = 3;
        a[1] = vector<double>(3); a[1][0] = 2; a[1][1] = 3; a[1][2] = 4;
        a[2] = vector<double>(3); a[2][0] = 5; a[2][1] = 6; a[2][2] = 7;

        vector<vector<double> > b(3, vector<double>(3));
        b[0] = vector<double>(3); b[0][0] = 3; b[0][1] = 2; b[0][2] = 1;
        b[1] = vector<double>(3); b[1][0] = 4; b[1][1] = 3; b[1][2] = 2;
        b[2] = vector<double>(3); b[2][0] = 5; b[2][1] = 4; b[2][2] = 3;
        
        vector<vector<double> > result = matrixMul(a, b);
    }

    // Test Case 2
    {
        vector<vector<double> > a(3, vector<double>(2));
        a[0] = vector<double>(2); a[0][0] = 0; a[0][1] = 0;
        a[1] = vector<double>(2); a[1][0] = 2; a[1][1] = 4;
        a[2] = vector<double>(2); a[2][0] = 1; a[2][1] = 2;

        vector<vector<double> > b(2, vector<double>(2));
        b[0] = vector<double>(2); b[0][0] = 0; b[0][1] = 0;
        b[1] = vector<double>(2); b[1][0] = 2; b[1][1] = 4;
        
        vector<vector<double> > result = matrixMul(a, b);
    }
    
    // Test Case 3
    {
        vector<vector<double> > a(3, vector<double>(2));
        a[0] = vector<double>(2); a[0][0] = 0; a[0][1] = 0;
        a[1] = vector<double>(2); a[1][0] = 2; a[1][1] = 4;
        a[2] = vector<double>(2); a[2][0] = 1; a[2][1] = 2;

        vector<vector<double> > b(3, vector<double>(3));
        b[0] = vector<double>(3); b[0][0] = 0; b[0][1] = 0; b[0][2] = 1;
        b[1] = vector<double>(3); b[1][0] = 2; b[1][1] = 4; b[1][2] = 1;
        b[2] = vector<double>(3); b[2][0] = 1; b[2][1] = 2; b[2][2] = 3;
        
        vector<vector<double> > result = matrixMul(a, b);
        if (result.empty()) {
            cout << "Matrix multiplication failed as expected." << endl;
        }
    }
}

int main()
{
    testMatrixMul();
    cout << "All matrixMul tests completed." << endl;
    return 0;
}
