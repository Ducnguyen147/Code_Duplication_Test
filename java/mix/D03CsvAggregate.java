public class D03CsvAggregate {
    public static java.util.Map<String, Double> aggregate(java.util.List<String> lines) {
        java.util.Map<String, Double> acc = new java.util.HashMap<>();
        for (String line : lines) {
            String[] parts = line.trim().split(",");
            if (parts.length != 2) continue;
            String key = parts[0];
            double val = Double.parseDouble(parts[1]);
            acc.put(key, acc.getOrDefault(key, 0.0) + val);
        }
        return acc;
    }
}
