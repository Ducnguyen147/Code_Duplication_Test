public class ProcessDataRenamed {
public static java.util.List<Integer> processDataV1(java.util.List<Integer> dataValues) {
    java.util.List<Integer> output = new java.util.ArrayList<>();
    for (int value : dataValues) {
        if (value % 2 == 0) {
            output.add(value * 2);
        }
    }
    return output;
}

}
