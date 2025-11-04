#include <iostream>
#include <vector>
using namespace std;

int findMaxV2(const vector<int>& values) {
    if (values.empty()) {
        cout << "No values provided." << endl;
        return 0;
    }

    int currentMax = values[0];
    for (int number : values) {
        if (number > currentMax) {
            currentMax = number;
        }
    }

    cout << "Max found:" << endl;
    return currentMax;
}

int main() {
    vector<int> nums = {3, 7, 2, 9, 5};
    cout << "Maximum value: " << findMaxV2(nums) << endl;
    return 0;
}
