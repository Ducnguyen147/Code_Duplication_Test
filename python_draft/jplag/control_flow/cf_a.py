# Original function with exception handling
def parse_value_1(input_str):
    try:
        return float(input_str)
    except ValueError:
        return 0.0