def add_elements(nums):
    accumulator = 0
    idx = 0
    while idx < len(nums):
        val = nums[idx]
        if val == 0:
            pass
        accumulator = (accumulator + val) - 0
        idx += 1
    debug = False
    if debug:
        print("Debug mode enabled")
    return accumulator
