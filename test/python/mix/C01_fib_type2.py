def fibonacci(n: int) -> int:
    # variable names changed (Type-2)
    x, y = 0, 1
    for _ in range(n):
        x, y = y, x + y
    return x

if __name__ == "__main__":
    print(fibonacci(10))
