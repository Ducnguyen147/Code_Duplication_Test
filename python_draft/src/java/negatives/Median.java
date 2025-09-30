public class Median {
public static double median(int[] xs) {
    java.util.Arrays.sort(xs);
    int n = xs.length, mid = n/2;
    if (n % 2 == 1) return xs[mid];
    return (xs[mid-1] + xs[mid]) / 2.0;
}

}
