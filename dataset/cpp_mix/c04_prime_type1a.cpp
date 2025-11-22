#include <iostream>
using namespace std;

bool isPrime(int n) {
    if (n < 2) {
        return false;
    }
    if (n % 2 == 0) {
        return n == 2;
    }
    int i = 3;
    while (i * i <= n) {
        if (n % i == 0) {
            return false;
        }
        i += 2;
    }
    return true;
}

int main() {
    cout << boolalpha << isPrime(10) << endl;
    return 0;
}
