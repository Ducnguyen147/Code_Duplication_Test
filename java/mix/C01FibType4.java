public class C01FibType4 {
    private static java.util.Map<Integer, Long> memo = new java.util.HashMap<>();
    public static long fib(int n) {
        if (n < 2) return n;
        Long hit = memo.get(n);
        if (hit != null) return hit;
        long res = fib(n - 1) + fib(n - 2);
        memo.put(n, res);
        return res;
    }
}
