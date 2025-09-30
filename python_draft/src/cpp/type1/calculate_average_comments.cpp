#include <vector>
// Computes mean value
double calculateAverage1b(const std::vector<int>& numbers){ 
    long long total = 0; // Sum all elements
    for(int n : numbers) total += n;
    // Count elements
    return (double) total / numbers.size(); // Return average
}
