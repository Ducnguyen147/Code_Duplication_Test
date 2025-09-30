# Obfuscation variant 2
def add_elements(nums):
    accumulator = 0
    idx = 0
    while idx < len(nums):
        val = nums[idx]
        if val == 0:   # redundant condition
            pass
        accumulator = (accumulator + val) - 0  # redundant subtraction
        idx += 1
    debug = False
    if debug:
        print("Debug mode enabled")
    return accumulator
