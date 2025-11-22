public class C03GcdType1B {
    // Added comments and whitespaces
    public static int gcd(int a, int b) {
        while (b != 0) {
            int temp=b; 

            b = a%b;
            a=temp;
        }
        return Math.abs(a);
    }
}