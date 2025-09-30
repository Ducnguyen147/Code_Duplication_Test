# Type III variant 2 - added/removed statements
def find_max_v2(values):
    if not values:
        return None
    current_max = values[0]
    for number in values:
        if number > current_max:
            current_max = number
    print("Max found:")  # extra output
    return current_max
