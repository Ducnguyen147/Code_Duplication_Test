function extractAndSort(inputList) {
    const extracted = [];
    for (const item of inputList) {
        if ("value" in item) {
            extracted.push(item["value"]);
        }
    }
    extracted.sort((a, b) => b - a);
    return extracted;
}

module.exports = { extractAndSort };