#include <vector>
double calculateAverage1a(   const std::vector<int>& numbers ){
    long long total=0; for(int n: numbers){ total+=n; } return (double)total/numbers.size();
}
