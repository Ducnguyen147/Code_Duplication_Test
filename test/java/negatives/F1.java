import java.util.List;

public class F1 {
    public static int processData(List<Object> inputList) {
        return helper(inputList);
    }

    private static int helper(List<Object> lst) {
        int total = 0;
        for (Object x : lst) {
            if (x instanceof List) {
                total += helper((List<Object>) x);
            } else if (x instanceof Integer) {
                total += (Integer) x;
            }
        }
        return total;
    }
}