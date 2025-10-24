import java.util.HashMap;
import java.util.Map;

public class C06WordcountType2 {
    public static Map<String, Integer> tallyWords(String text) {
        Map<String, Integer> tally = new HashMap<>();
        for (String token : text.split("\\s+")) {
            tally.put(token, tally.getOrDefault(token, 0) + 1);
        }
        return tally;
    }
}

