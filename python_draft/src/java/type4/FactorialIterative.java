public class FactorialIterative {
public static long factorialV1(int num) {
    long result = 1L;
    for (int i=1; i<=num; i++) result *= i;
    return result;
}

}
