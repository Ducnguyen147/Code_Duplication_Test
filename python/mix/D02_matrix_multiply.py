def matmul(A, B):
    m, n, p = len(A), len(A[0]), len(B[0])
    assert len(B) == n
    C = [[0]*p for _ in range(m)]
    for i in range(m):
        for k in range(n):
            aik = A[i][k]
            for j in range(p):
                C[i][j] += aik * B[k][j]
    return C
