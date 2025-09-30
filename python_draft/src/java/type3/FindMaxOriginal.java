public class FindMaxOriginal {
public static int findMax(int[] values) {
    int maxVal = values[0];
    for (int i=1;i<values.length;i++) {
        if (values[i] > maxVal) maxVal = values[i];
    }
    return maxVal;
}

}
