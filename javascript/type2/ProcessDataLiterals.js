function processDataV2(inputList) {
    const result = [];
    for (const item of inputList) {
        if (item % 4 === 0) {
            result.push(item * 3);
        }
    }
    return result;
}

module.exports = { processDataV2 };
