public class C04PrimeType4 {
    public static boolean isPrime(int n) {
        if (n < 2) return false;
        if (n % 2 == 0) return n == 2;
        for (int d = 3; ; d += 2) {
            int q = n / d;
            if (q < d) break;
            if (n % d == 0) return false;
        }
        return true;
    }
}
