function processData(inputList) {
    return helper(inputList);
}

function helper(lst) {
    let total = 0;
    for (const x of lst) {
        if (Array.isArray(x)) {
            total += helper(x);
        } else if (typeof x === "number") {
            total += x;
        }
    }
    return total;
}

module.exports = { processData };
