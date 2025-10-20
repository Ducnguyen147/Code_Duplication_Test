def process_data(input_list):
    def recursion_sum(lst):
        total = 0
        for x in lst:
            if isinstance(x, list):
                total += helper(x)
            elif isinstance(x, int):
                total += x
        return total

    return helper(input_list)
