# Original implementation
def factorial_1(n):
    if n == 0:
        return 1
    else:
        return n * factorial_1(n-1)