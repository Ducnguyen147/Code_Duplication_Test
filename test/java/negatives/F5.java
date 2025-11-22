import java.util.*;

public class F5 {
    public static List<Integer> extractAndSort(List<Map<String, Integer>> inputList) {
        List<Integer> extracted = new ArrayList<>();
        for (Map<String, Integer> item : inputList) {
            if (item.containsKey("value")) {
                extracted.add(item.get("value"));
            }
        }
        extracted.sort(Comparator.reverseOrder());
        return extracted;
    }
}
