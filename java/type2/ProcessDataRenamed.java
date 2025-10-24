import java.util.*;

public class ProcessDataRenamed {
    public static List<Integer> processDatav1(List<Integer> dataValues) {
        List<Integer> output = new ArrayList<>();
        for (int value : dataValues) {
            if (value % 2 == 0) {
                output.add(value * 2);
            }
        }
        return output;
    }
}