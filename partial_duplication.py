# partial_duplication.py

def calculate_area(length, width):
    area = length * width
    return area

def calculate_perimeter(length, width):
    perimeter = 2 * (length + width)
    return perimeter

# Partially duplicated logic in the function
def calculate_area_again(length, width):
    area = length * width  # Same code as calculate_area
    # Additional logic
    print(f"The area is: {area}")
    return area
