function sumList(arr){
  let total = 0;
  for(const x of arr) total += x;
  return total;
}
module.exports = { sumList };
