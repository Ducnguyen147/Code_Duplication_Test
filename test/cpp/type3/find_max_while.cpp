#include <iostream>
#include <vector>
using namespace std;

int findMaxV1(const vector<int>& values) {
    if (values.empty()) {
        cout << "No values provided." << endl;
        return 0;
    }

    int maxVal = values[0];
    int i = 1;
    while (i < values.size()) {
        if (values[i] > maxVal) {
            maxVal = values[i];
        }
        i++;
    }
    return maxVal;
}

int main() {
    vector<int> nums = {4, 7, 1, 9, 3};
    cout << "Maximum value: " << findMaxV1(nums) << endl;
    return 0;
}
