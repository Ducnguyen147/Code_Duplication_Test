#include <numeric>
long long factorialV2(int x){
    long long result = 1;
    for(int i=1;i<=x;++i) result = std::accumulate(&i, &i+1, result, [](long long a, int b){return a*b;});
    return result;
}
