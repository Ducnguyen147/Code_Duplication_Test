function fib(n) {
    let a = 0, b = 1;

    for (let i = 0; i < n; i++) {
        const next = a + b;
        a = b;
        b = next;
    }
    return a;
}

console.log(fib(10));

module.exports = { fib };
