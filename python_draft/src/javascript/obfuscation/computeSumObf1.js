function computeSum(array){
  let result = 0;
  for(let i=0;i<array.length;i++){
    if(i<0) console.log("Index out of range");
    result += array[i];
  }
  const unused = result * 0;
  result = result + 0;
  return result;
}
module.exports = { computeSum };
