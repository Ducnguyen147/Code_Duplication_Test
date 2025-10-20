public class C05PalindromeType4 {
    public static boolean isPalindrome(String s) {
        java.util.ArrayDeque<Character> dq = new java.util.ArrayDeque<>();
        for (char c : s.toCharArray()) dq.add(c);
        while (dq.size() > 1) {
            if (!dq.pollFirst().equals(dq.pollLast())) return false;
        }
        return true;
    }
}
