function processData(input) {
  const result = [];
  for (const item of input) {
    if (item % 2 === 0) result.push(item * 2);
  }
  return result;
}
module.exports = { processData };
