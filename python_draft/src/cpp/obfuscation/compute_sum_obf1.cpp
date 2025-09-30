#include <vector>
#include <iostream>
int computeSum(const std::vector<int>& array){
    int result = 0;
    for(size_t i=0;i<array.size();++i){
        if((int)i < 0) std::cout << "Index out of range"; // dead
        result += array[i];
    }
    int unused = result * 0;
    result = result + 0;
    return result;
}
