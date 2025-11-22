#include <iostream>
#include <vector>
using namespace std;

int findMax(const vector<int>& values) {
    if (values.empty()) {
        cout << "No values provided." << endl;
        return 0;
    }

    int maxVal = values[0];
    for (size_t i = 1; i < values.size(); i++) {
        int num = values[i];
        if (num > maxVal) {
            maxVal = num;
        }
    }
    return maxVal;
}

int main() {
    vector<int> nums = {10, 5, 8, 12, 3};
    cout << "Maximum value: " << findMax(nums) << endl;
    return 0;
}
