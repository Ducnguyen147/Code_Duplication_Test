function factorial(n) {
    if (n <= 0) return 1;

    return Array.from({ length: n }, (_, i) => i + 1)
        .reduce((a, b) => a * b, 1);
}

console.log(factorial(5));
module.exports = { factorial };
