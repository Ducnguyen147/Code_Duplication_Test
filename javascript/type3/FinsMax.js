function findMax(values) {
    let maxVal = values[0];
    for (let i = 1; i < values.length; i++) {
        const num = values[i];
        if (num > maxVal) {
            maxVal = num;
        }
    }
    return maxVal;
}

module.exports = { findMax };
