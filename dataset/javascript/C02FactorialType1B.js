function fact(n) {
    let result = 1; // same as r
    for (let i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

console.log(fact(5));

module.exports = { fact };
