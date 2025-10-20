class CalculateAverageComments {
    // Computes mean value over an array
    static double calculateAverage(double[] numbers) {
        // Sum all elements
        double total = 0;
        for (double x : numbers) total += x;
        // Count elements
        int count = numbers.length;
        // Return average
        return total / count;
    }
}