function findMaxV1(values) {
    let maxVal = values[0];
    let i = 1;
    while (i < values.length) {
        if (values[i] > maxVal) {
            maxVal = values[i];
        }
        i++;
    }
    return maxVal;
}

module.exports = { findMaxV1 };
