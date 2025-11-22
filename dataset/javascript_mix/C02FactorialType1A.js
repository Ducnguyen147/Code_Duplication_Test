function fact(n) {
    let r = 1;
    for (let i = 2; i <= n; i++) {
        r *= i;
    }
    return r;
}

console.log(fact(5));

module.exports = { fact };
