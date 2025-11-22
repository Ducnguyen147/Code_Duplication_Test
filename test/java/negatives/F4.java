import java.util.List;
import java.util.stream.Collectors;

public class F4 {
    public static int productOfEvens(List<Integer> inputList) {
        List<Integer> doubled = inputList.stream()
                .filter(x -> x % 2 == 0)
                .map(x -> x * 2)
                .collect(Collectors.toList());

        return doubled.stream()
                .reduce(1, (a, b) -> a * b);
    }
}
