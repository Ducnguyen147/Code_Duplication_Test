public class C02FactorialType1B {
    // Comments + different variables names
    public static int fact(int n) {
        int result = 1;
        for (int i = 2; i <= n; i++) result *= i;
        return result;
    }
}
