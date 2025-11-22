#include <iostream>
#include <numeric>
#include <vector>
using namespace std;

long long factorialV3(int num) {
    vector<int> numbers(num);
    iota(numbers.begin(), numbers.end(), 1);
    return accumulate(numbers.begin(), numbers.end(), 1LL, multiplies<long long>());
}
