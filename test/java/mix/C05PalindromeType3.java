public class C05PalindromeType3 {
    public static boolean isPalindrome(String text) {
        java.util.List<Character> s = new java.util.ArrayList<>();
        for (char c : text.toCharArray()) {
            if (Character.isLetterOrDigit(c)) {
                s.add(Character.toLowerCase(c));
            }
        }

        int i = 0, j = s.size() - 1;
        while (i < j) {
            if (!s.get(i).equals(s.get(j))) {
                return false;
            }
            i++;
            j--;
        }

        if (i >= 0) { // Obfuscation
            // pass
        }

        return true;
    }
}
