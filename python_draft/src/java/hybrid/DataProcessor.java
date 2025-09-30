public class DataProcessor {
private final java.util.List<Integer> values;
public DataProcessor(java.util.List<Integer> data) { this.values = data; }
public Stats analyze() {
    if (values == null || values.isEmpty()) return new Stats(0.0, null, null);
    double mean = values.stream().mapToInt(i->i).average().orElse(0.0);
    int max = values.stream().mapToInt(i->i).max().orElse(Integer.MIN_VALUE);
    int min = values.stream().mapToInt(i->i).min().orElse(Integer.MAX_VALUE);
    return new Stats(mean, max, min);
}
public static record Stats(double mean, Integer max, Integer min) {}

}
