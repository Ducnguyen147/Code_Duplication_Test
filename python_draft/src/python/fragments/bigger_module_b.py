# Mixed functions module B
def helper_sum(xs):
    s = 0
    for z in xs:
        s += z
    return s

def calculate_average_1a(    numbers ):
    total=helper_sum(numbers)
    count= len( numbers)
    return total/count

def unrelated(a, b):
    return a*b - a - b

def another():
    return "ok"
