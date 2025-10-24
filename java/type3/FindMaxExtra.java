public class FindMaxExtra {
    public static Integer findMaxV2(int[] values) {
        if (values == null || values.length == 0) {
            return null;
        }
        int currentMax = values[0];
        for (int number : values) {
            if (number > currentMax) {
                currentMax = number;
            }
        }
        System.out.println("Max found:");
        return currentMax;
    }
}