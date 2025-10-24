public class C05PalindromeType1B {
    // Added comments and white spaces
    public static boolean isPalindrome(String text) {
        StringBuilder cleaned = new StringBuilder();
        
        for (char ch : text.toCharArray()) {
            if (Character.isLetterOrDigit(ch)) {
                    cleaned.append(Character.toLowerCase(ch));
            }
        }

        String cleanedStr = cleaned.toString();
        return cleanedStr.equals(new StringBuilder(cleanedStr).reverse().toString());
    }
}
