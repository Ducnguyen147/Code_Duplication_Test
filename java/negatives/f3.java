import java.util.*;
class N3 {
    static Map<Integer,Integer> countModulo(List<Integer> in){
        Map<Integer,Integer> m = new HashMap<>();
        for (int x : in){
            int r = x % 3;
            m.put(r, m.getOrDefault(r, 0) + 1);
        }
        return m;
    }
}