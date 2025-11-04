#include <iostream>
#include <string>
#include <map>
#include <sstream>
#include <vector>
using namespace std;

map<string, double> aggregateLines(const vector<string>& lines) {
    map<string, double> acc;

    for (const string& line : lines) {
        string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));
        trimmed.erase(trimmed.find_last_not_of(" \t\n\r") + 1);

        stringstream ss(trimmed);
        string key, valStr;

        if (getline(ss, key, ',') && getline(ss, valStr)) {
            double val = stod(valStr);
            acc[key] += val;
        }
    }

    return acc;
}

int main() {
    vector<string> lines = {
        "apple,10.5",
        "banana,5.0",
        "apple,4.5",
        "banana,3.5"
    };

    map<string, double> result = aggregateLines(lines);

    for (const auto& [key, val] : result) {
        cout << key << ": " << val << endl;
    }

    return 0;
}
