import java.util.stream.IntStream;

public class FactorialFunctional {
    public static int factorial_v2(int x) {
        return IntStream.rangeClosed(1, x)
                        .reduce(1, (a, b) -> a * b);
    }
}