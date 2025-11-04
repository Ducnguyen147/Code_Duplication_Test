#include <iostream>
#include <vector>
#include <variant>
using namespace std;

using NestedList = vector<variant<int, vector<variant<int, vector<variant<int, vector<variant<int, vector<int>>>>>>>>>;

int helper(const vector<variant<int, vector<variant<int, vector<variant<int, vector<variant<int, vector<int>>>>>>>>> &lst) {
    int total = 0;
    for (const auto &x : lst) {
        if (holds_alternative<int>(x)) {
            total += get<int>(x);
        } else if (holds_alternative<vector<variant<int, vector<variant<int, vector<variant<int, vector<variant<int, vector<int>>>>>>>>> >(x)) {
            total += helper(get<vector<variant<int, vector<variant<int, vector<variant<int, vector<variant<int, vector<int>>>>>>>>> >(x));
        }
    }
    return total;
}

int processData(const vector<variant<int, vector<variant<int, vector<variant<int, vector<variant<int, vector<int>>>>>>>>> &inputList) {
    return helper(inputList);
}

int main() {
    // Example: [1, [2, 3], [4, [5]]]
    vector<variant<int, vector<variant<int, vector<variant<int, vector<variant<int, vector<int>>>>>>>>> inputList = {
        1,
        vector<variant<int, vector<variant<int, vector<variant<int, vector<variant<int, vector<int>>>>>>>>>{
            2, 3
        },
        vector<variant<int, vector<variant<int, vector<variant<int, vector<variant<int, vector<int>>>>>>>>>{
            4,
            vector<variant<int, vector<variant<int, vector<variant<int, vector<variant<int, vector<int>>>>>>>>>{
                5
            }
        }
    };

    cout << "Total: " << processData(inputList) << endl;
    return 0;
}
