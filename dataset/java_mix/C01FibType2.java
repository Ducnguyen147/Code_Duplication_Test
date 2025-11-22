public class C01FibType2 {
    // Variables name change
    public static int fib(int n) {
        int x = 0, y = 1;
        int k = 0;
        while (k < n) {
            int t = x;
            x = y;
            y = t + y;
            k++;
        }
        return x;
    }

    public static void main(String[] args) {
        System.out.println(fib(10));
    }
}
