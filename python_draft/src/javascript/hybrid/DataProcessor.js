class DataProcessor {
  constructor(data){ this.values = data; }
  analyze(){
    if(!this.values || this.values.length===0) return {mean:0,max:null,min:null};
    const mean = this.values.reduce((a,b)=>a+b,0)/this.values.length;
    return {mean, max: Math.max(...this.values), min: Math.min(...this.values)};
  }
}
module.exports = { DataProcessor };
