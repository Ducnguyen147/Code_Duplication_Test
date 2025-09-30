# Mixed functions module A
import math

def helper_round(x):
    return round(x, 3)

def calculate_average_1(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / count

def unrelated(a, b):
    return a**2 + b**2

def format_report(values):
    return f"n={len(values)} avg={calculate_average_1(values):.3f}"
