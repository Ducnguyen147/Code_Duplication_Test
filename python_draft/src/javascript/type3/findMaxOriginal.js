function findMax(values) {
  let maxVal = values[0];
  for (let i = 1; i < values.length; i++) {
    if (values[i] > maxVal) maxVal = values[i];
  }
  return maxVal;
}
module.exports = { findMax };
