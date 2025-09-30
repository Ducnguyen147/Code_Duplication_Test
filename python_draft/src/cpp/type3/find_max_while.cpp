#include <vector>
int findMaxV1(const std::vector<int>& values){
    int maxVal = values[0];
    size_t i=1;
    while(i<values.size()){
        if(values[i] > maxVal) maxVal = values[i];
        ++i;
    }
    return maxVal;
}
