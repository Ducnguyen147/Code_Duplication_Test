# Original control structure
def find_max_1(values):
    max_val = values[0]
    for num in values[1:]:
        if num > max_val:
            max_val = num
    return max_val

# Type III variant 1 - changed loop type
def find_max_2(values):
    max_val = values[0]
    i = 1
    while i < len(values):
        if values[i] > max_val:
            max_val = values[i]
        i += 1
    return max_val

# Type III variant 2 - added/removed statements
def find_max_3(values):
    if not values:
        return None
    current_max = values[0]
    for number in values:
        if number > current_max:
            current_max = number
    print("Max found:")
    return current_max