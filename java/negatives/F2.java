import java.util.*;

public class F2 {
    public static List<Integer> squaresPositive(List<Integer> inputList){
        List<Integer> out = new ArrayList<>();
        for (int x : inputList) if (x >= 0) out.add(x*x);
        return out;
    }
}