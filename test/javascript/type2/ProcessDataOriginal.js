function processData(inputList) {
    const result = [];
    for (const item of inputList) {
        if (item % 2 === 0) {
            result.push(item * 2);
        }
    }
    return result;
}

module.exports = { processData };
