#include <iostream>
using namespace std;

int fact(int n) {
    int result = 1; // same as r
    for (int i=2;i<=n;i++) {
        result *= i;
    }
    return result;
}

int main() {
    cout << fact(5) << endl;
    return 0;
}
