#include <vector>
#include <algorithm>
std::vector<int> processData(const std::vector<int>& input){
    std::vector<int> result;
    for(int item : input){
        if(item % 2 == 0) result.push_back(item * 2);
    }
    return result;
}
