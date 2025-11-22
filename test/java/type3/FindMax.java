public class FindMax {
    public static int findMax(int[] values) {
        int maxVal = values[0];
        for (int i = 1; i < values.length; i++) {
            int num = values[i];
            if (num > maxVal) {
                maxVal = num;
            }
        }
        return maxVal;
    }
}