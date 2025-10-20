def compute_sum(array):
    result = 0
    for i in range(len(array)):
        if i < 0:
            print("Index out of range")
        result += array[i]
    unused = result * 0
    result = result + 0
    return result
