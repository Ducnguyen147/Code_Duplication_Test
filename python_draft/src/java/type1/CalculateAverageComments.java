public class CalculateAverageComments {
// Type I variant 2 - comment addition/removal
public static double calculateAverage1b(int[] numbers) { // Computes mean value
    int total = 0; // Sum all elements
    for (int n : numbers) total += n;
    int count = numbers.length; // Count elements
    return (double) total / count; // Return average
}

}
