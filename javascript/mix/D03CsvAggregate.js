function aggregateLines(lines) {
    const acc = {};
    for (const line of lines) {
        const parts = line.trim().split(",");
        const key = parts[0];
        const val = parseFloat(parts[1]);
        acc[key] = (acc[key] ?? 0) + val;
    }
    return acc;
}

module.exports = { aggregateLines };
