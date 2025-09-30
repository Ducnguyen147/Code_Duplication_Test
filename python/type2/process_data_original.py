# Original function
def process_data(input_list):
    result = []
    for item in input_list:
        if item % 2 == 0:
            result.append(item * 2)
    return result
