#include <iostream>
#include <vector>
#include <map>
using namespace std;

map<int, int> countModulo(const vector<int>& inputList) {
    map<int, int> freq;
    for (int x : inputList) {
        int r = x % 3;
        freq[r]++;
    }
    return freq;
}

int main() {
    vector<int> nums = {3, 4, 5, 6, 7, 8, 9};
    map<int, int> result = countModulo(nums);

    for (const auto& [remainder, count] : result) {
        cout << "Remainder " << remainder << ": " << count << endl;
    }

    return 0;
}
