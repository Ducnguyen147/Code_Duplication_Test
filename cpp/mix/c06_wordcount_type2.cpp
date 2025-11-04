#include <iostream>
#include <sstream>
#include <map>
#include <string>
using namespace std;

map<string, int> tallyWords(const string& text) {
    map<string, int> tally;
    istringstream iss(text);
    string token;

    while (iss >> token) {
        tally[token]++;
    }

    return tally;
}

int main() {
    string text = "apple banana apple orange banana apple";
    map<string, int> result = tallyWords(text);

    for (const auto& [word, count] : result) {
        cout << word << ": " << count << endl;
    }

    return 0;
}
