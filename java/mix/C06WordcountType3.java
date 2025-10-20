public class C06WordcountType3 {
    public static java.util.Map<String,Integer> wordCount(String s) {
        java.util.Map<String,Integer> m = new java.util.HashMap<>();
        String[] arr = s.split("\\s+");
        for (int i = 0; i < arr.length; i++) {
            String w = arr[i];
            Integer v = m.get(w);
            m.put(w, (v == null ? 1 : v + 1));
        }
        return m;
    }
}
