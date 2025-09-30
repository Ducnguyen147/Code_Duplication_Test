# Type IV variant 2 - functional approach
from functools import reduce
def factorial_3(x):
    return reduce(lambda a,b: a*b, range(1, x+1), 1)