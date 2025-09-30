# Original control structure
def find_max(values):
    max_val = values[0]
    for num in values[1:]:
        if num > max_val:
            max_val = num
    return max_val
