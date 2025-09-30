def reverse_string(s):
    loop_reversed = ""
    for char in s:
        loop_reversed = char + loop_reversed
    return loop_reversed