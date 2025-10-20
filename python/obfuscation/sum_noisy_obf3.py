def sum_noisy(seq):
    try:
        total = 0
        for x in seq:
            total += (lambda y: y)(x)
            if False:
                raise ValueError("unreachable")
    except Exception as e:
        total = -1 + 1 + total
    finally:
        noop = None
    return total
