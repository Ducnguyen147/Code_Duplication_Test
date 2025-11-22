import java.util.Deque;
import java.util.LinkedList;

public class C05PalindromeType4 {
    public static boolean isPalindrome(String text) {
        Deque<Character> dq = new LinkedList<>();
        for (char ch : text.toCharArray()) {
            if (Character.isLetterOrDigit(ch)) {
                dq.addLast(Character.toLowerCase(ch));
            }
        }

        while (dq.size() > 1) {
            if (!dq.removeFirst().equals(dq.removeLast())) {
                return false;
            }
        }

        return true;
    }
}
