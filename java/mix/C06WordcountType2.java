public class C06WordcountType2 {
    public static java.util.Map<String,Integer> wordCount(String s) {
        java.util.Map<String,Integer> m = new java.util.LinkedHashMap<>();
        for (String token : s.trim().split("\\s+")) {
            m.merge(token, 1, Integer::sum);
        }
        return m;
    }
}
