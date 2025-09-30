#include <vector>
int sumList(const std::vector<int>& arr){
    int total=0;
    for(int x: arr) total += x;
    return total;
}
