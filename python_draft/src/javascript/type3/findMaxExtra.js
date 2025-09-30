function findMaxV2(values) {
  if (!values || values.length === 0) return null;
  let currentMax = values[0];
  for (const number of values) {
    if (number > currentMax) currentMax = number;
  }
  console.log("Max found:"); // extra
  return currentMax;
}
module.exports = { findMaxV2 };
