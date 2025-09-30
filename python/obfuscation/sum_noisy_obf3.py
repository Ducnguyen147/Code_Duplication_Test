# Obfuscation variant 3
def sum_noisy(seq):
    # Shadowing, pointless try/except, and no-op branches
    try:
        total = 0
        for x in seq:
            total += (lambda y: y)(x)
            if False:
                raise ValueError("unreachable")
    except Exception as e:
        # Will never execute
        total = -1 + 1 + total
    finally:
        noop = None
    return total
