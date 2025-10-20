import java.util.*;
class FactorialFunctional {
    static long factorialV2(int n) {
        return java.util.stream.IntStream.rangeClosed(1, n).asLongStream().reduce(1L, (a,b)->a*b);
    }
}