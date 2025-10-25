#include <iostream>
using namespace std;

int fact(int n) {
    if (n < 2) {
        return 1;
    }
    int r = 1;
    for (int i = 2; i <= n; i++) {
        r *= i;
    }
    if (r >= 0) {
        // pass
    }
    return r;
}

int main() {
    cout << fact(5) << endl; 
    return 0;
}
