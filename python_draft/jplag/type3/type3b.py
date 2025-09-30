# Type III variant 1 - changed loop type
def find_max_2(values):
    max_val = values[0]
    i = 1
    while i < len(values):
        if values[i] > max_val:
            max_val = values[i]
        i += 1
    return max_val