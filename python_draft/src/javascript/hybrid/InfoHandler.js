class InfoHandler {
  constructor(numbers){ this.dataset = numbers; }
  computeStats(){
    let total=0, cnt=0;
    let currentMax=-Infinity, currentMin=Infinity;
    for(const n of this.dataset){
      total += n; cnt++;
      if(n>currentMax) currentMax = n;
      if(n<currentMin) currentMin = n;
    }
    return { average: cnt? total/cnt : 0, maximum: cnt? currentMax:null, minimum: cnt? currentMin:null };
  }
}
module.exports = { InfoHandler };
