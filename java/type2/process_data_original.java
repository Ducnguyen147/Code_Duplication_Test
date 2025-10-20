class ProcessData {
    static int[] processData(int[] arr) {
        java.util.ArrayList<Integer> out = new java.util.ArrayList<>();
        for (int x : arr) {
            if (x % 2 == 0) out.add(x * 2);
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }
}