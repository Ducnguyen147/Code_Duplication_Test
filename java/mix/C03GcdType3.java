public class C03GcdType3 {
    public static int gcd(int a, int b) {
        a = Math.abs(a); b = Math.abs(b);
        if (a == 0) return b;
        if (b == 0) return a;
        while (a != b) {
            if (a > b) a -= b;
            else b -= a;
        }
        return a;
    }
}
