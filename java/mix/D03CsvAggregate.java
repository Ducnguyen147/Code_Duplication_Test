import java.util.HashMap;
import java.util.Map;

public class D03CsvAggregate {
    public static Map<String, Double> aggregateLines(Iterable<String> lines) {
        Map<String, Double> acc = new HashMap<>();
        for (String line : lines) {
            String[] parts = line.strip().split(",");
            String key = parts[0];
            double val = Double.parseDouble(parts[1]);
            acc.put(key, acc.getOrDefault(key, 0.0) + val);
        }
        return acc;
    }
}
