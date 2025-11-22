function gcd(a, b) {
    if (a === 0) {
        return Math.abs(b);
    }
    if (b === 0) {
        return Math.abs(a);
    }
    while (b !== 0) {
        const temp = b;
        b = a % b;
        a = temp;
    }
    return Math.abs(a);
}

module.exports = { gcd };
