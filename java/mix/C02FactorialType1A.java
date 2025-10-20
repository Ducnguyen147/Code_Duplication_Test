public class C02FactorialType1A {
    public static long fact(int n) {
        long r = 1;
        for (int i = 2; i <= n; i++) r *= i;
        return r;
    }
}
