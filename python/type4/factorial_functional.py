from functools import reduce
def factorial_v2(x):
    return reduce(lambda a,b: a*b, range(1, x+1), 1)
