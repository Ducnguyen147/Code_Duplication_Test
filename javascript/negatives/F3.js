function countModulo(inputList) {
    const freq = {};
    for (const x of inputList) {
        const r = x % 3;
        freq[r] = (freq[r] ?? 0) + 1;
    }
    return freq;
}

module.exports = { countModulo };
