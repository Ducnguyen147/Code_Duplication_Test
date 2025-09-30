#include <vector>
int findMax(const std::vector<int>& values){
    int maxVal = values[0];
    for(size_t i=1;i<values.size();++i){
        if(values[i] > maxVal) maxVal = values[i];
    }
    return maxVal;
}
