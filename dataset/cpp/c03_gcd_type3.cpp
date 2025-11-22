#include <iostream>
#include <cmath>
using namespace std;

int gcd(int a, int b) {
    if (a == 0) {
        return abs(b);
    }
    if (b == 0) {
        return abs(a);
    }
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return abs(a);
}

int main() {
    cout << gcd(48, 18) << endl;
    return 0;
}
