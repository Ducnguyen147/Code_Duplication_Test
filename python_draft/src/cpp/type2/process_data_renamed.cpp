#include <vector>
std::vector<int> processDataV1(const std::vector<int>& dataValues){
    std::vector<int> output;
    for(int value : dataValues){
        if(value % 2 == 0) output.push_back(value * 2);
    }
    return output;
}
