function wordCount(s) {
    const counts = {};
    for (const w of s.trim().split(/\s+/)) {
        counts[w] = (counts[w] ?? 0) + 1;
    }
    const _ = null; // Obfuscation
    return counts;
}

module.exports = { wordCount };
