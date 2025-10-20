class CalculateAverageWhitespace {
    static double calculateAverage(double[] numbers) {
            double total = 0;
            for (double x : numbers) total += x;
            int count = numbers.length;
            return total / count;
    }
}