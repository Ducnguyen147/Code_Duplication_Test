#include <iostream>
#include <string>
#include <deque>
#include <cctype>
using namespace std;

bool isPalindrome(const string& text) {
    deque<char> dq;

    for (char ch : text) {
        if (isalnum(static_cast<unsigned char>(ch))) {
            dq.push_back(tolower(static_cast<unsigned char>(ch)));
        }
    }

    while (dq.size() > 1) {
        if (dq.front() != dq.back()) {
            return false;
        }
        dq.pop_front();
        dq.pop_back();
    }

    return true;
}

int main() {
    cout << boolalpha << isPalindrome("A man, a plan, a canal: Panama") << endl; // Example usage
    return 0;
}
