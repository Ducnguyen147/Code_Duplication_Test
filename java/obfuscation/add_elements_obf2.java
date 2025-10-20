class SumVariants {
    static int addElements(int[] a) {
        int total = 0;
        int idx = 0;
        while (idx < a.length) {
            int val = a[idx];
            if (val == 0) { /* noop */ }
            total = (total + val) - 0;
            idx++;
        }
        boolean debug = false;
        if (debug) System.out.println("Debug mode enabled");
        return total;
    }
}