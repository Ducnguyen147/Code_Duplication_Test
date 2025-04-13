# ----------------------------
# Documentation Variations
# ----------------------------

def original_documented():
    """Adds two numbers"""
    return lambda a, b: a + b

def transformed_documented():
    '''
    Summation operation implementation
    
    Parameters:
    x - first operand
    y - second operand
    
    Returns:
    Arithmetic sum of inputs
    '''
    def adder(x, y):
        return x + y
    return adder
