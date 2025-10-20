import java.util.*;
class N4 {
    static long productOfEvens(List<Integer> in){
        long acc = 1;
        for (int x : in) if (x % 2 == 0) acc *= (long)(x * 2);
        return acc;
    }
}