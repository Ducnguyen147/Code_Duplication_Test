public class C04PrimeType1B {
    public static boolean isPrime(int n) {
        if (n < 2) return false;
        if (n % 2 == 0) return n == 2;
        int limit = (int)Math.sqrt(n);
        for (int d = 3; d <= limit; d += 2) {
            if (n % d == 0) return false;
        }
        return true;
    }
}
