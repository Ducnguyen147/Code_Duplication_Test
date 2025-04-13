# Original implementation
def factorial_1(n):
    if n == 0:
        return 1
    else:
        return n * factorial_1(n-1)

# Type IV variant 1 - iterative approach
def factorial_2(num):
    result = 1
    for i in range(1, num+1):
        result *= i
    return result

# Type IV variant 2 - functional approach
from functools import reduce
def factorial_3(x):
    return reduce(lambda a,b: a*b, range(1, x+1), 1)