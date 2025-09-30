# Type II variant 1 - renamed identifiers
def handle_info(data_values):
    output = []
    for value in data_values:
        if value % 2 == 0:
            output.append(value * 2)
    return output