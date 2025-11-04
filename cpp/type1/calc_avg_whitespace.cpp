#include <iostream>
#include <vector>
#include <numeric>
using namespace std;

double calculateAverageWhitespace(const vector<int>& numbers) {
    if (numbers.empty()) return 0.0;

    int total = accumulate(numbers.begin(), numbers.end(), 0);
    
    int count = numbers.size();
    return static_cast<double>(total)/count;

}

int main() {
    vector<int> nums = {10,20,30,40,50};
    cout << "Average: " << calculateAverageOriginal(nums) << endl;
    return 0;
}
