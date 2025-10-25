function matmul(A, B) {
    const m = A.length;
    const n = A[0].length;
    const p = B[0].length;
    console.assert(B.length === n, "Matrix dimension mismatch");

    const C = Array.from({ length: m }, () => Array(p).fill(0));

    for (let i = 0; i < m; i++) {
        for (let k = 0; k < n; k++) {
            const aik = A[i][k];
            for (let j = 0; j < p; j++) {
                C[i][j] += aik * B[k][j];
            }
        }
    }
    return C;
}

module.exports = { matmul };
