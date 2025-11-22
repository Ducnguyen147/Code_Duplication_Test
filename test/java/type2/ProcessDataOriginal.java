import java.util.*;

public class ProcessDataOriginal {
    public static List<Integer> processData(List<Integer> inputList) {
        List<Integer> result = new ArrayList<>();
        for (int item : inputList) {
            if (item % 2 == 0) {
                result.add(item * 2);
            }
        }
        return result;
    }
}