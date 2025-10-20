public class C03GcdType2 {
    public static int gcd(int a, int b) {
        a = Math.abs(a); b = Math.abs(b);
        if (b == 0) return a;
        return gcd(b, a % b);
    }
}
