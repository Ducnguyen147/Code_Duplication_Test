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