class SumVariants {
    static int sumNoisy(int[] a) {
        int total = 0;
        for (int x : a) {
            total += ((java.util.function.IntUnaryOperator)(y->y)).applyAsInt(x);
            if (false) throw new RuntimeException("unreachable");
        }
        return total;
    }
}