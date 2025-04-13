# Original function
def process_data(input_list):
    result = []
    for item in input_list:
        if item % 2 == 0:
            result.append(item * 2)
    return result

# Type II variant 1 - renamed identifiers
def handle_info(data_values):
    output = []
    for value in data_values:
        if value % 2 == 0:
            output.append(value * 2)
    return output

# Type II variant 2 - changed literals
def process_data_v2(input_list):
    result = []
    for item in input_list:
        if item % 4 == 0:
            result.append(item * 3)
    return result