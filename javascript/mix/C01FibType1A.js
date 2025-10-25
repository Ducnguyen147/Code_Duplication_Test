function fib(n) {
    let a = 0, b = 1;
    for (let i = 0; i < n; i++) {
        const temp = a;
        a = b;
        b = temp + b;
    }
    return a;
}

console.log(fib(10));

module.exports = { fib };
