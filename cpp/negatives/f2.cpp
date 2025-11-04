#include <iostream>
#include <vector>
using namespace std;

vector<int> squaresPositive(const vector<int>& inputList) {
    vector<int> out;
    for (int x : inputList) {
        if (x >= 0) {
            out.push_back(x * x);
        }
    }
    return out;
}

int main() {
    vector<int> nums = {-3, -1, 0, 2, 4};
    vector<int> result = squaresPositive(nums);

    for (int val : result) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}
