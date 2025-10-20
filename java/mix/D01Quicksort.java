public class D01Quicksort {
    public static java.util.List<Integer> quicksort(java.util.List<Integer> a) {
        if (a.size() <= 1) return new java.util.ArrayList<>(a);
        int pivot = a.get(a.size() / 2);
        java.util.List<Integer> left = new java.util.ArrayList<>();
        java.util.List<Integer> mid  = new java.util.ArrayList<>();
        java.util.List<Integer> right= new java.util.ArrayList<>();
        for (int x : a) {
            if (x < pivot) left.add(x);
            else if (x > pivot) right.add(x);
            else mid.add(x);
        }
        left = quicksort(left);
        right = quicksort(right);
        left.addAll(mid);
        left.addAll(right);
        return left;
    }
}
