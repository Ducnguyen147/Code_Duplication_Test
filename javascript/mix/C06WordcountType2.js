function tallyWords(text) {
    const tally = {};
    const tokens = text.trim().split(/\s+/);
    for (const token of tokens) {
        tally[token] = (tally[token] ?? 0) + 1;
    }
    return tally;
}

module.exports = { tallyWords };
