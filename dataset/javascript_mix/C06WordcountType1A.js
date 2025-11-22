function wordCount(s) {
    const counts = {};
    const words = s.trim().split(/\s+/);
    for (const w of words) {
        counts[w] = (counts[w] ?? 0) + 1;
    }
    return counts;
}

module.exports = { wordCount };
