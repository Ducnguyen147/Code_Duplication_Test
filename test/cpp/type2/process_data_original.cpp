#include <iostream>
#include <vector>
using namespace std;

vector<int> processData(const vector<int>& inputList) {
    vector<int> result;
    for (int item : inputList) {
        if (item % 2 == 0) {
            result.push_back(item * 2);
        }
    }
    return result;
}

int main() {
    vector<int> nums = {1, 2, 3, 4, 5, 6};
    vector<int> result = processData(nums);

    cout << "Processed values: ";
    for (int val : result) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}
