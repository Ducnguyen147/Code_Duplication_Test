#include <iostream>
#include <cmath>
using namespace std;

int gcd(int x, int y) {
    while (y != 0) {
        int temp = y;
        y = x % y;
        x = temp;
    }
    return abs(x);
}

int main() {
    cout << gcd(48, 18) << endl;
    return 0;
}
