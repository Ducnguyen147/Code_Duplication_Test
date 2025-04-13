# Original function with standard formatting
def calculate_average_1(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / count

# Type I variant 1 - whitespace changes
def calculate_average_1a(    numbers ):
    total=sum(numbers)
    count= len( numbers)
    return total/count

# Type I variant 2 - comment addition/removal
def calculate_average_1b(numbers):  # Computes mean value
    total = sum(numbers)  # Sum all elements
    count = len(numbers)  # Count elements
    return total / count  # Return average