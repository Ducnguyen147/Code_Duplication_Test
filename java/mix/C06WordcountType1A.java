import java.util.HashMap;
import java.util.Map;

public class C06WordcountType1A {
    public static Map<String, Integer> wordCount(String s) {
        Map<String, Integer> counts = new HashMap<>();
        for (String w : s.split("\\s+")) {
            counts.put(w, counts.getOrDefault(w, 0) + 1);
        }
        return counts;
    }
}
