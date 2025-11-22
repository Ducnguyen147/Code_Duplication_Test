// Computes mean value
function calculateAverage1b(numbers) {
    const total = numbers.reduce((a, b) => a + b, 0); // Sum all elements
    const count = numbers.length; // Count elements
    return total / count; // Return average
}

module.exports = { calculateAverage1b };
