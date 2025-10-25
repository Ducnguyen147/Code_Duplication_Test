function gcd(x, y) {
    while (y !== 0) {
        const temp = y;
        y = x % y;
        x = temp;
    }
    return Math.abs(x);
}

module.exports = { gcd };