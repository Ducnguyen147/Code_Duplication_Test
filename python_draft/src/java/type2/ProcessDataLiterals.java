public class ProcessDataLiterals {
public static java.util.List<Integer> processDataV2(java.util.List<Integer> input) {
    java.util.List<Integer> result = new java.util.ArrayList<>();
    for (int item : input) {
        if (item % 4 == 0) {
            result.add(item * 3);
        }
    }
    return result;
}

}
