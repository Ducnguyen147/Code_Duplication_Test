public class D02MatrixMultiply {
    public static int[][] matmul(int[][] A, int[][] B) {
        int m = A.length;
        int n = A[0].length;
        int p = B[0].length;
        assert B.length == n;

        int[][] C = new int[m][p];

        for (int i = 0; i < m; i++) {
            for (int k = 0; k < n; k++) {
                int aik = A[i][k];
                for (int j = 0; j < p; j++) {
                    C[i][j] += aik * B[k][j];
                }
            }
        }
        return C;
    }
}
