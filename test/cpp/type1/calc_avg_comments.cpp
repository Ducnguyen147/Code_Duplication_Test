#include <iostream>
#include <vector>
#include <numeric> // for accumulate
using namespace std;

// Computes mean value
double calculateAverage1b(const vector<int>& numbers) {
    if (numbers.empty()) return 0.0;

    int total = accumulate(numbers.begin(), numbers.end(), 0); // Sum all elements
    int count = numbers.size(); // Count elements
    return static_cast<double>(total) / count; // Return average
}

int main() {
    vector<int> nums = {2, 4, 6, 8, 10};
    cout << "Average: " << calculateAverage1b(nums) << endl;
    return 0;
}
