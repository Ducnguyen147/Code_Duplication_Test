public class AddElementsObf2 {
public static int addElements(int[] nums) {
    int acc = 0; int idx = 0;
    while (idx < nums.length) {
        int val = nums[idx];
        if (val == 0) {} // redundant
        acc = (acc + val) - 0;
        idx++;
    }
    boolean debug = false;
    if (debug) System.out.println("Debug mode enabled");
    return acc;
}

}
