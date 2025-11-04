#include <iostream>
using namespace std;

long long factorialRecursive(int n) {
    if (n == 0) {
        return 1;
    }
    return n * factorialRecursive(n - 1);
}
