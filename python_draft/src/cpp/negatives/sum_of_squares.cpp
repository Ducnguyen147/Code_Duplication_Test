#include <vector>
int sumOfSquares(const std::vector<int>& xs){
    int s = 0; for(int v: xs) s += v*v; return s;
}
