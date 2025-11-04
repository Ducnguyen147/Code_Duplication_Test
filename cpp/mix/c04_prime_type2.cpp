#include <iostream>
using namespace std;

bool prime(int n) {
    if (n < 2) {
        return false;
    }
    if (n % 2 == 0) {
        return n == 2;
    }
    int k = 3;
    while (k * k <= n) {
        if (n % k == 0) {
            return false;
        }
        k += 2;
    }
    return true;
}

int main() {
    cout << boolalpha << prime(17) << endl;
    return 0;
}
