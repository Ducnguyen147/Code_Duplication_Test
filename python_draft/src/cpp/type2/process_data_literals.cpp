#include <vector>
std::vector<int> processDataV2(const std::vector<int>& input){
    std::vector<int> result;
    for(int item : input){
        if(item % 4 == 0) result.push_back(item * 3);
    }
    return result;
}
