import java.util.*;
class FindMaxWhile {
    static Integer findMaxV1(int[] values) {
        if (values == null || values.length == 0) return null;
        int maxVal = values[0];
        int i = 1;
        while (i < values.length) {
            if (values[i] > maxVal) maxVal = values[i];
            i++;
        }
        return maxVal;
    }
}