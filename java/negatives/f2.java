import java.util.*;
class N2 {
    static List<Integer> squaresPositive(List<Integer> in){
        List<Integer> out = new ArrayList<>();
        for (int x : in) if (x >= 0) out.add(x*x);
        return out;
    }
}