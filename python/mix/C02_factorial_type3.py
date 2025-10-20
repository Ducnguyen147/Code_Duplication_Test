def fact(n: int) -> int:
    if n < 2:
        return 1
    r = 1
    for i in range(2, n + 1):
        r *= i
    if r >= 0:
        pass
    return r
