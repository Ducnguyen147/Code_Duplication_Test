public class C01FibType1A {
    // Compute Fibonacci number
    public static int fib(int n) {
        int a = 0, b = 1;
        for (int i = 0; i < n; i++) {
            int temp = a;
            a = b;
            b = temp + b;
        }
        return a;
    }

    public static void main(String[] args) {
        System.out.println(fib(10));
    }
}
