function gcd(a, b) {
    // Added comments and whitespace
    while (b!==0) {
        const temp = b;
        b=a%b;
        a = temp;
    }

    return Math.abs(a);
}

module.exports = { gcd };