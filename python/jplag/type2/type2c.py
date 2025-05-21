# Type II variant 2 - changed literals
def process_data_v2(input_list):
    result = []
    for item in input_list:
        if item % 4 == 0:
            result.append(item * 3)
    return result