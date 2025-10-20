class ProcessDataRenamed {
    static int[] processDataV1(int[] arr) {
        java.util.ArrayList<Integer> out = new java.util.ArrayList<>();
        for (int value : arr) {
            if (value % 2 == 0) out.add(value * 2);
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }
}