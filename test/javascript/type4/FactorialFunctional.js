function factorialV2(x) {
    return Array.from({ length: x }, (_, i) => i + 1)
        .reduce((a, b) => a * b, 1);
}

module.exports = { factorialV2 };
