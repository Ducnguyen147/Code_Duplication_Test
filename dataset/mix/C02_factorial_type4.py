import math

def factorial(n: int) -> int:
    return math.prod(range(1, n + 1)) if n > 0 else 1
