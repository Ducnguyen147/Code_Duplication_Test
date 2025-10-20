public class C02FactorialType3 {
    public static long fact(int n) {
        long r = 1;
        int i = 2;
        while (i <= n) {
            r *= i;
            i++;
        }
        return r;
    }
}
