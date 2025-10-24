import java.util.stream.IntStream;

public class C02FactorialType4 {
    public static int factorial(int n) {
        return n > 0 ? IntStream.rangeClosed(1, n).reduce(1, (a, b) -> a * b) : 1;
    }
}
