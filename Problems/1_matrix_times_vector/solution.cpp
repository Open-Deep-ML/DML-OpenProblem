#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

vector<double> matrixDotVector(const vector<vector<double> >& a,
                               const vector<double>& b) 
{
    // Handle the empty case
    if (a.empty() && b.empty()) {
        vector<double> empty;
        return empty;
    }

    // If one is empty or dimensions do not match, return invalid
    if (a.empty() || b.empty() || a[0].size() != b.size()) {
        // Return a vector of size 1 containing -1.0 to signal error
        vector<double> invalid(1, -1.0);
        return invalid;
    }

    // Perform the multiplication
    vector<double> c(a.size(), 0.0);
    for (int i = 0; i < (int)a.size(); i++) {
        for (int j = 0; j < (int)a[i].size(); j++) {
            c[i] += a[i][j] * b[j];
        }
    }
    return c;
}

void testMatrixDotVector()
{
    // Empty product
    {
        vector<vector<double> > a; 
        vector<double> b;         
        vector<double> result = matrixDotVector(a, b);
        assert(result.empty());
    }

    // Invalid Products
    {
        vector<vector<double> > a; 
        vector<double> b(2);
        b[0] = 1.0;
        b[1] = 2.0;
        vector<double> result = matrixDotVector(a, b);
        assert(result.size() == 1 && result[0] == -1.0);
    }
    {
        vector<vector<double> > a;
        vector<double> row1(2);
        row1[0] = 1.0;
        row1[1] = 2.0;
        a.push_back(row1);

        vector<double> b;
        vector<double> result = matrixDotVector(a, b);
        assert(result.size() == 1 && result[0] == -1.0);
    }
    {
        // Dimension mismatch
        vector<vector<double> > a;
        {
            vector<double> row1(2);
            row1[0] = 1.0;
            row1[1] = 2.0;
            a.push_back(row1);

            vector<double> row2(2);
            row2[0] = 2.0;
            row2[1] = 4.0;
            a.push_back(row2);
        }

        vector<double> b(1);
        b[0] = 1.0;

        vector<double> result = matrixDotVector(a, b);
        assert(result.size() == 1 && result[0] == -1.0);
    }

    // Valid product
    {
        vector<vector<double> > a;
        {
            vector<double> row1(2);
            row1[0] = 1.0;
            row1[1] = 2.0;
            a.push_back(row1);

            vector<double> row2(2);
            row2[0] = 2.0;
            row2[1] = 4.0;
            a.push_back(row2);
        }

        vector<double> b(2);
        b[0] = 1.0;
        b[1] = 2.0;

        vector<double> result = matrixDotVector(a, b);
        assert(result.size() == 2 && result[0] == 5.0 && result[1] == 10.0);
    }

    {
        vector<vector<double> > a;
        {
            vector<double> row1(3);
            row1[0] = 1.0;
            row1[1] = 2.0;
            row1[2] = 3.0;
            a.push_back(row1);

            vector<double> row2(3);
            row2[0] = 2.0;
            row2[1] = 4.0;
            row2[2] = 6.0;
            a.push_back(row2);
        }

        vector<double> b(3);
        b[0] = 1.0;
        b[1] = 2.0;
        b[2] = 3.0;

        vector<double> result = matrixDotVector(a, b);
        assert(result.size() == 2 && result[0] == 14.0 && result[1] == 28.0);
    }
}

int main()
{
    testMatrixDotVector();
    cout << "All tests passed." << endl;
    return 0;
}
