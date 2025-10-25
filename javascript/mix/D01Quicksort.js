function quicksort(a) {
    if (a.length <= 1) {
        return [...a];
    }

    const pivot = a[Math.floor(a.length / 2)];
    const left = [];
    const mid = [];
    const right = [];

    for (const x of a) {
        if (x < pivot) {
            left.push(x);
        } else if (x === pivot) {
            mid.push(x);
        } else {
            right.push(x);
        }
    }

    return [...quicksort(left), ...mid, ...quicksort(right)];
}

module.exports = { quicksort };
