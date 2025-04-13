# ----------------------------
# Advanced Type IV Transformations
# ----------------------------

# Original matrix multiplication
def matrix_mult_1(a, b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]

# Semantic equivalent with different implementation
def matrix_mult_2(m1, m2):
    result = []
    for i in range(len(m1)):
        result_row = []
        for j in range(len(m2[0])):
            total = 0
            for k in range(len(m2)):
                total += m1[i][k] * m2[k][j]
            result_row.append(total)
        result.append(result_row)
    return result