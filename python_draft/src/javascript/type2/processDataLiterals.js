function processDataV2(input) {
  const result = [];
  for (const item of input) {
    if (item % 4 === 0) result.push(item * 3);
  }
  return result;
}
module.exports = { processDataV2 };
