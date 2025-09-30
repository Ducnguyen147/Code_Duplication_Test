public class FactorialRecursive {
public static long factorial(int n) {
    if (n == 0) return 1L;
    return n * factorial(n-1);
}

}
