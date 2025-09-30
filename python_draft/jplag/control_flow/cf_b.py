# Variant with conditional validation
def parse_value_2(text):
    if text.replace('.', '', 1).isdigit():
        return float(text)
    else:
        return 0.0