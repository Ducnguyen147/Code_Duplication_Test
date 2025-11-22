#include <iostream>
using namespace std;

long long factorialV1(int num) {
    long long result = 1;
    for (int i = 1; i <= num; i++) {
        result *= i;
    }
    return result;
}