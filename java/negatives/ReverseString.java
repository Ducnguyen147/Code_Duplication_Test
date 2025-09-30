public class ReverseString {
    public static String reverse(String s) {
        String loopReversed = "";
        for (int i = 0; i < s.length(); i++) {
            char currentChar = s.charAt(i);
            loopReversed = currentChar + loopReversed;
        }
        return loopReversed;
    }
}
