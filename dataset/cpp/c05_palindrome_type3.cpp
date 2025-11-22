#include <iostream>
#include <string>
#include <vector>
#include <cctype>
using namespace std;

bool isPalindrome(const string& text) {
    vector<char> s;
    for (char c : text) {
        if (isalnum(static_cast<unsigned char>(c))) {
            s.push_back(tolower(static_cast<unsigned char>(c)));
        }
    }

    int i = 0, j = static_cast<int>(s.size()) - 1;
    while (i < j) {
        if (s[i] != s[j]) {
            return false;
        }
        i++;
        j--;
    }

    if (i >= 0) { // Obfuscation
        // pass
    }

    return true;
}

int main() {
    cout << boolalpha << isPalindrome("RaceCar") << endl;
    return 0;
}
