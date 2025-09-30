// Type I variant 2 - comments
function calculateAverage1b(numbers) { // Computes mean value
  const total = numbers.reduce((a,b)=>a+b, 0); // Sum elements
  const count = numbers.length; // Count elements
  return total / count; // Return average
}
module.exports = { calculateAverage1b };
