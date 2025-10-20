public class C01FibType2 {
    public static long fib(int n) {
        long a = 0, b = 1;
        int k = 0;
        while (k < n) {
            long t = a;
            a = b;
            b = t + b;
            k++;
        }
        return a;
    }
}
