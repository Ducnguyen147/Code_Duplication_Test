#include <iostream>
using namespace std;

// Variables name change
int fib(int n) {
    int x = 0, y = 1;
    int k = 0;
    while (k < n) {
        int t = x;
        x = y;
        y = t + y;
        k++;
    }
    return x;
}

int main() {
    cout << fib(10) << endl;
    return 0;
}
