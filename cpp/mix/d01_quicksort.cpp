#include <iostream>
#include <vector>
using namespace std;

vector<int> quicksort(const vector<int>& a) {
    if (a.size() <= 1) {
        return a;
    }

    int pivot = a[a.size() / 2];
    vector<int> left, mid, right;

    for (int x : a) {
        if (x < pivot) {
            left.push_back(x);
        } else if (x == pivot) {
            mid.push_back(x);
        } else {
            right.push_back(x);
        }
    }

    vector<int> sortedLeft = quicksort(left);
    vector<int> sortedRight = quicksort(right);

    vector<int> result;
    result.reserve(sortedLeft.size() + mid.size() + sortedRight.size());
    result.insert(result.end(), sortedLeft.begin(), sortedLeft.end());
    result.insert(result.end(), mid.begin(), mid.end());
    result.insert(result.end(), sortedRight.begin(), sortedRight.end());

    return result;
}

int main() {
    vector<int> arr = {3, 6, 8, 10, 1, 2, 1};
    vector<int> sorted = quicksort(arr);

    for (int x : sorted) {
        cout << x << " ";
    }
    cout << endl;

    return 0;
}
