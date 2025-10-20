public class C04PrimeType2 {
    public static boolean isPrime(int n) {
        if (n < 2) return false;
        int d = 2;
        while (d * d <= n) {
            if (n % d == 0) return false;
            d += (d == 2) ? 1 : 2; // skip even numbers after 2
        }
        return true;
    }
}
