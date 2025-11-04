#include <iostream>
#include <vector>
using namespace std;

vector<int> processDatav1(const vector<int>& dataValues) {
    vector<int> output;
    for (int value : dataValues) {
        if (value % 2 == 0) {
            output.push_back(value * 2);
        }
    }
    return output;
}

int main() {
    vector<int> data = {1, 2, 3, 4, 5, 6};
    vector<int> result = processDatav1(data);

    cout << "Processed values: ";
    for (int val : result) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}
