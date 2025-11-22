import java.util.*;

public class ProcessDataLiterals {
    public static List<Integer> processDataV2(List<Integer> inputList) {
        List<Integer> result = new ArrayList<>();
        for (int item : inputList) {
            if (item % 4 == 0) {
                result.add(item * 3);
            }
        }
        return result;
    }
}