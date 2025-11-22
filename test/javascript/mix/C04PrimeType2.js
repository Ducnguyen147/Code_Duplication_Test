function prime(n) {
    if (n < 2) {
        return false;
    }
    if (n % 2 === 0) {
        return n === 2;
    }
    let k = 3;
    while (k * k <= n) {
        if (n % k === 0) {
            return false;
        }
        k += 2;
    }
    return true;
}

module.exports = { prime };
