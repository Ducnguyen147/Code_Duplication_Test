public class InfoHandler {
    private final java.util.List<Integer> dataset;
    public InfoHandler(java.util.List<Integer> numbers) { this.dataset = numbers; }
    public Stats computeStats() {
        int total = 0, cnt = 0;
        int currentMax = Integer.MIN_VALUE, currentMin = Integer.MAX_VALUE;
        for (int n : dataset) {
            total += n; cnt++;
            if (n > currentMax) currentMax = n;
            if (n < currentMin) currentMin = n;
        }
        double avg = cnt==0 ? 0.0 : (double)total / cnt;
        return new Stats(avg, currentMax, currentMin);
    }
    public static record Stats(double average, Integer maximum, Integer minimum) {}
}
