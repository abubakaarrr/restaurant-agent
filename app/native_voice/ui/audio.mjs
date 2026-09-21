export function pcm16(chunks, inputRate, outputRate=24000){
  const length=chunks.reduce((n,c)=>n+c.length,0), source=new Float32Array(length);
  let offset=0;for(const c of chunks){source.set(c,offset);offset+=c.length;}
  const count=Math.floor(length*outputRate/inputRate), bytes=new ArrayBuffer(count*2), view=new DataView(bytes), ratio=inputRate/outputRate;
  for(let i=0;i<count;i++){
    const begin=Math.floor(i*ratio),end=Math.min(length,Math.max(begin+1,Math.floor((i+1)*ratio)));
    let sum=0;for(let j=begin;j<end;j++)sum+=source[j];
    const sample=Math.max(-1,Math.min(1,sum/(end-begin)));
    view.setInt16(i*2,Math.round(sample*(sample<0?32768:32767)),true);
  }
  return bytes;
}

export class VoiceActivity {
  constructor(rate){this.rate=rate;this.reset();this.pre=[];this.preDuration=0;}
  reset(){this.chunks=[];this.active=false;this.voiceMs=0;this.quietMs=0;this.totalMs=0;this.triggered=false;}
  feed(samples,{threshold=.018,silenceMs=1400}={}){
    const ms=samples.length/this.rate*1000;
    const rms=Math.sqrt(samples.reduce((s,x)=>s+x*x,0)/samples.length);
    const voiced=rms>=threshold;
    if(!this.active){
      this.pre.push(samples);this.preDuration+=ms;
      while(this.preDuration>240 && this.pre.length>1)this.preDuration-=this.pre.shift().length/this.rate*1000;
      if(!voiced)return {rms};
      this.active=true;this.chunks=this.pre;this.pre=[];this.totalMs=this.preDuration;this.preDuration=0;
    }else{this.chunks.push(samples);this.totalMs+=ms;}
    if(voiced){this.voiceMs+=ms;this.quietMs=0;}else this.quietMs+=ms;
    const started=!this.triggered&&this.voiceMs>=180;
    if(started)this.triggered=true;
    if(this.quietMs>=silenceMs || this.totalMs>=59000){
      const chunks=this.voiceMs>=250?this.chunks:null;
      const quietMs=this.quietMs;
      this.reset();return {rms,started,chunks,quietMs,ended:true};
    }
    return {rms,started,voiced};
  }
}
