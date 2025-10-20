public class C06WordcountType1B {
    public static java.util.Map<String,Integer> wordCount(String s) {
        java.util.Map<String,Integer> m = new java.util.HashMap<>();
        for (String w : s.split("\\s+")) {
            m.put(w, m.getOrDefault(w, 0) + 1);
        }
        return m;
    }
}
