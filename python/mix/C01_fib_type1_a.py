def fib(n: int) -> int:
    """Return the n-th Fibonacci number (0-indexed)."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

if __name__ == "__main__":
    print(fib(10))
