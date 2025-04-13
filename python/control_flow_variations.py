# ----------------------------
# Control Flow Variations
# ----------------------------

# Original error handling
def parse_value_1(input_str):
    try:
        return float(input_str)
    except ValueError:
        return 0.0

# Type III/IV variant - structural/semantic change
def parse_value_2(text):
    if text.replace('.', '', 1).isdigit():
        return float(text)
    else:
        return 0.0