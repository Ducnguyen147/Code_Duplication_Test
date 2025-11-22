#include <iostream>
#include <vector>
using namespace std;

int productOfEvens(const vector<int>& inputList) {
    vector<int> doubled;
    for (int x : inputList) {
        if (x % 2 == 0) {
            doubled.push_back(x * 2);
        }
    }

    int product = 1;
    for (int val : doubled) {
        product *= val;
    }

    return product;
}

int main() {
    vector<int> nums = {1, 2, 3, 4, 5, 6};
    cout << "Product of doubled evens: " << productOfEvens(nums) << endl;
    return 0;
}
