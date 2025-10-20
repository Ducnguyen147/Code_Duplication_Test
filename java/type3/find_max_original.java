import java.util.*;
class FindMax {
    static Integer findMax(int[] values) {
        if (values == null || values.length == 0) return null;
        int maxVal = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > maxVal) maxVal = values[i];
        }
        return maxVal;
    }
}