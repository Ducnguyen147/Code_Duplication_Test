public class SumListOriginal {
    public static int sumList(int[] arr) {
        int total = 0;
        for (int x : arr) total += x;
        return total;
    }
}
