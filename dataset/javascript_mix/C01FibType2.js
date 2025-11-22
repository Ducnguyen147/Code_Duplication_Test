// Variables name change
function fib(n) {
    let x = 0, y = 1;
    let k = 0;
    while (k < n) {
        const t = x;
        x = y;
        y = t + y;
        k++;
    }
    return x;
}

console.log(fib(10));

module.exports = { fib };
