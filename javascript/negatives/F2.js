function squaresPositive(inputList) {
    const out = [];
    for (const x of inputList) {
        if (x >= 0) {
            out.push(x * x);
        }
    }
    return out;
}

module.exports = { squaresPositive };
