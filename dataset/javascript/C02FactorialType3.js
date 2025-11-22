function fact(n) {
    if (n < 2) {
        return 1;
    }
    let r = 1;
    for (let i = 2; i <= n; i++) {
        r *= i;
    }
    if (r >= 0) {
        // pass
    }
    return r;
}

console.log(fact(5));

module.exports = { fact };
