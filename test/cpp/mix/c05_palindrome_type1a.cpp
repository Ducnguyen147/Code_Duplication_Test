#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>
using namespace std;

bool isPalindrome(const string& text) {
    string cleaned;
    for (char ch : text) {
        if (isalnum(static_cast<unsigned char>(ch))) {
            cleaned += tolower(static_cast<unsigned char>(ch));
        }
    }

    string reversed = cleaned;
    reverse(reversed.begin(), reversed.end());
    return cleaned == reversed;
}

int main() {
    cout << boolalpha << isPalindrome("A man, a plan, a canal: Panama") << endl; // Example usage
    return 0;
}
