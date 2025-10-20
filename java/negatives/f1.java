class N1 {
    static int recursion_sum(Object[] lst) {
        int total = 0;
        for (Object x : lst) {
            if (x instanceof Object[]) total += helper((Object[])x);
            else if (x instanceof Integer) total += (Integer)x;
        }
        return total;
    }
    static int processData(Object[] input) {
        return helper(input);
    }
}