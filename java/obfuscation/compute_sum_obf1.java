class SumVariants {
    static int computeSum(int[] a) {
        int total = 0;
        for (int i = 0; i < a.length; i++) {
            if (i < 0) System.out.println("Index out of range");
            total += a[i];
        }
        int unused = total * 0;
        total = total + 0;
        return total;
    }
}