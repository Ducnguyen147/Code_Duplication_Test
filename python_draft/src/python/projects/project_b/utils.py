def clamp_value(x, lo, hi):  # different util, same behavior
    if x < lo: return lo
    if x > hi: return hi
    return x
