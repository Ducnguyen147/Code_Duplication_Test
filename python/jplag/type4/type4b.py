# Type IV variant 1 - iterative approach
def factorial_2(num):
    result = 1
    for i in range(1, num+1):
        result *= i
    return result