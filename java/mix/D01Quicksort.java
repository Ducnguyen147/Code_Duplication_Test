import java.util.ArrayList;
import java.util.List;

public class D01Quicksort {
    public static List<Integer> quicksort(List<Integer> a) {
        if (a.size() <= 1) {
            return new ArrayList<>(a);
        }

        int pivot = a.get(a.size() / 2);
        List<Integer> left = new ArrayList<>();
        List<Integer> mid = new ArrayList<>();
        List<Integer> right = new ArrayList<>();

        for (int x : a) {
            if (x < pivot) {
                left.add(x);
            } else if (x == pivot) {
                mid.add(x);
            } else {
                right.add(x);
            }
        }

        List<Integer> sorted = new ArrayList<>();
        sorted.addAll(quicksort(left));
        sorted.addAll(mid);
        sorted.addAll(quicksort(right));
        return sorted;
    }
}
