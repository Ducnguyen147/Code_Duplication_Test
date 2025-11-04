#include <iostream>
#include <vector>
using namespace std;

vector<int> processDataV2(const vector<int>& inputList) {
    vector<int> result;
    for (int item : inputList) {
        if (item % 4 == 0) {
            result.push_back(item * 3);
        }
    }
    return result;
}

int main() {
    vector<int> nums = {2, 4, 8, 10, 12};
    vector<int> result = processDataV2(nums);

    cout << "Processed values: ";
    for (int val : result) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}
