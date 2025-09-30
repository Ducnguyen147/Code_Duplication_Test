public class ProcessDataOriginal {
public static java.util.List<Integer> processData(java.util.List<Integer> input) {
    java.util.List<Integer> result = new java.util.ArrayList<>();
    for (int item : input) {
        if (item % 2 == 0) {
            result.add(item * 2);
        }
    }
    return result;
}

}
