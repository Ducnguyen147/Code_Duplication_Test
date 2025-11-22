#include <iostream>
using namespace std;

int fact(int n) {
    int r = 1;
    for (int i = 2; i <= n; i++) {
        r *= i;
    }
    return r;
}

int main() {
    cout << fact(5) << endl;
    return 0;
}
