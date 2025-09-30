public class FindMaxWhile {
    public static int findMaxV1(int[] values) {
        int maxVal = values[0];
        int i = 1;
        while (i < values.length) {
            if (values[i] > maxVal) maxVal = values[i];
            i++;
        }
        return maxVal;
    }
}
