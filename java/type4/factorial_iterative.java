class FactorialIterative {
    static long factorialV1(int num) {
        long result = 1;
        for (int i = 1; i <= num; i++) result *= i;
        return result;
    }
}