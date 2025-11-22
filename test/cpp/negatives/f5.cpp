#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
using namespace std;

vector<int> extractAndSort(const vector<map<string, int>>& inputList) {
    vector<int> extracted;
    for (const auto& item : inputList) {
        auto it = item.find("value");
        if (it != item.end()) {
            extracted.push_back(it->second);
        }
    }

    sort(extracted.begin(), extracted.end(), greater<int>());
    return extracted;
}

int main() {
    vector<map<string, int>> data = {
        {{"value", 5}},
        {{"value", 2}},
        {{"other", 7}},
        {{"value", 9}}
    };

    vector<int> result = extractAndSort(data);

    cout << "Sorted extracted values (descending): ";
    for (int val : result) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}
