def fact(n: int) -> int:
    """Iterative factorial"""
    result=1  # same as r
    for i in range(2, n+1):
        result *= i
    return result
