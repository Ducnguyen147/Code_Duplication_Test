#include <iostream>
#include <sstream>
#include <map>
#include <string>
using namespace std;

map<string, int> wordCount(const string& s) {
    map<string, int> counts;
    istringstream iss(s);
    string word;
    while (iss >> word) {
        counts[word]++;
    }
    return counts;
}

int main() {
    string text = "this is a test this is only a test";
    map<string, int> result = wordCount(text);

    for (const auto& [word, count] : result) {
        cout << word << ": " << count << endl;
    }

    return 0;
}
