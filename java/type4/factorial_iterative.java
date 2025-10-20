import java.util.*;
class FactorialIterative {
    static long factorialV1(int n) {
        long r = 1;
        for (int i = 1; i <= n; i++) r *= i;
        return r;
    }
}