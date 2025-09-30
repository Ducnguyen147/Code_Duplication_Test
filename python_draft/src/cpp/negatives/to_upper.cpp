#include <string>
std::string toUpper(std::string s){
    for(char& c: s) if('a'<=c && c<='z') c = c - 'a' + 'A';
    return s;
}
