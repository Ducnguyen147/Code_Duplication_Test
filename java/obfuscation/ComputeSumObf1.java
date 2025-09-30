public class ComputeSumObf1 {
    public static int computeSum(int[] array) {
        int result = 0;
        for (int i=0;i<array.length;i++) {
            if (i < 0) System.out.println("Index out of range");
            result += array[i];
        }
        int unused = result * 0;
        result = result + 0;
        return result;
    }
}
