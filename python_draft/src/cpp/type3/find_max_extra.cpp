#include <vector>
#include <iostream>
int findMaxV2(const std::vector<int>& values){
    if(values.empty()) return -2147483648;
    int currentMax = values[0];
    for(int number : values){
        if(number > currentMax) currentMax = number;
    }
    std::cout << "Max found:" << std::endl;
    return currentMax;
}
