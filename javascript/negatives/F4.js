function productOfEvens(inputList) {
    const doubled = inputList
        .filter(x => x % 2 === 0)
        .map(x => x * 2);

    return doubled.reduce((a, b) => a * b, 1);
}

module.exports = { productOfEvens };
