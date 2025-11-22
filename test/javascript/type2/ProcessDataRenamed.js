function processDatav1(dataValues) {
    const output = [];
    for (const value of dataValues) {
        if (value % 2 === 0) {
            output.push(value * 2);
        }
    }
    return output;
}

module.exports = { processDatav1 };
