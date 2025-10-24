import java.util.*;

public class F3 {
    public static Map<Integer,Integer> countModulo(List<Integer> inputList){
        Map<Integer,Integer> freq = new HashMap<>();
        for (int x : inputList){
            int r = x % 3;
            freq.put(r, freq.getOrDefault(r, 0) + 1);
        }
        return freq;
    }
}