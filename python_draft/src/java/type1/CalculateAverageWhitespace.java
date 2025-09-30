public class CalculateAverageWhitespace {
// Type I variant 1 - whitespace changes
public static double calculateAverage1a(   int[] numbers){
    int total=0; for(int n: numbers){ total+=n; } int count = numbers.length ;
    return (double)total/count;
}

}
