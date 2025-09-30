public class CalculateAverageOriginal {
// Original function with standard formatting
public static double calculateAverage1(int[] numbers) {
    int total = 0;
    for (int n : numbers) total += n;
    int count = numbers.length;
    return (double) total / count;
}

}
