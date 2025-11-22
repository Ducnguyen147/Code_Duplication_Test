public class C04PrimeType1A {
    public static boolean isPrime(int n) {
        if (n < 2) {
            return false;
        }
        if (n % 2 == 0) {
            return n == 2;
        }
        int i = 3;
        while (i * i <= n) {
            if (n % i == 0) {
                return false;
            }
            i += 2;
        }
        return true;
    }
}
