#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

vector<vector<int>> matmul(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int m = A.size();
    int n = A[0].size();
    int p = B[0].size();

    assert(B.size() == n && "Matrix dimension mismatch");

    vector<vector<int>> C(m, vector<int>(p, 0));

    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            int aik = A[i][k];
            for (int j = 0; j < p; j++) {
                C[i][j] += aik * B[k][j];
            }
        }
    }

    return C;
}

int main() {
    vector<vector<int>> A = {{1, 2, 3}, {4, 5, 6}};
    vector<vector<int>> B = {{7, 8}, {9, 10}, {11, 12}};

    vector<vector<int>> C = matmul(A, B);

    for (const auto& row : C) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}
