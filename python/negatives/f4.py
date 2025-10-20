from functools import reduce
import operator

def product_of_evens(input_list):
    evens = filter(lambda x: x % 2 == 0, input_list)
    doubled = map(lambda x: x * 2, evens)
    try:
        return reduce(operator.mul, doubled, 1)
    except TypeError:
        return 1
