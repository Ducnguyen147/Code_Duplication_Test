function calculateAverage1a(    numbers) {
    const total = numbers.reduce((a, b) => a + b, 0);
    
    const count = numbers.length;
    return total / count;
}

module.exports = { calculateAverage1a };
