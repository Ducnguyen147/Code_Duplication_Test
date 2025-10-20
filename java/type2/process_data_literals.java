class ProcessDataLiterals {
    static int[] processDataV2(int[] arr) {
        java.util.ArrayList<Integer> out = new java.util.ArrayList<>();
        for (int item : arr) {
            if (item % 4 == 0) out.add(item * 3);
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }
}