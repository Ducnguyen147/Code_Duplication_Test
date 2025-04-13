# ----------------------------
# Nested Structure Variations
# ----------------------------

def original_nested():
    return [x**2 for x in range(10) if x % 2 == 0]

def transformed_nested():
    result = []
    for num in range(10):
        if num % 2 != 0:
            continue
        result.append(num * num)
    return result