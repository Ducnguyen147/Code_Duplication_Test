import java.util.*;
class FindMaxExtra {
    static Integer findMaxV2(int[] values) {
        if (values == null || values.length == 0) return null;
        int maxVal = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > maxVal) maxVal = values[i];
        }
        System.out.println("Max found:");
        return maxVal;
    }
}