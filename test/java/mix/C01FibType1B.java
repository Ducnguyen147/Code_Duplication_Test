public class C01FibType1B {
    //Compute Fibonacci number - Different comments, whitespaces
    public static int fib(int n) {
        int a = 0, b = 1;

        for (int i=0; i < n; i++) {
            int next = a + b;
            a=b;
            b = next;
        }
        return a;
    }
    public static void main(String[] args) {
        System.out.println(fib(10));
    }
}
