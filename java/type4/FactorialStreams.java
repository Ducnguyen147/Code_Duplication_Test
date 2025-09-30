public class FactorialStreams {
    public static long factorialV2(int x) {
        return java.util.stream.IntStream.rangeClosed(1, x).asLongStream().reduce(1L, (a,b)->a*b);
    }
}
