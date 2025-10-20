import java.util.*;
class N5 {
    static List<Integer> extractAndSort(List<Map<String,Integer>> in){
        List<Integer> out = new ArrayList<>();
        for (Map<String,Integer> m : in) if (m.containsKey("value")) out.add(m.get("value"));
        out.sort(java.util.Collections.reverseOrder());
        return out;
    }
}