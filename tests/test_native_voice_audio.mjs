import assert from 'node:assert/strict';
import {pcm16,VoiceActivity} from '../app/native_voice/ui/audio.mjs';
const rate=48000, tone=new Float32Array(2048).fill(.1), quiet=new Float32Array(2048);
const vad=new VoiceActivity(rate);let started=0,ended=0,total=0;
// One continuous 30-second message must stay one turn, including short pauses.
for(let i=0;i<700;i++){
  const r=vad.feed(i%150>140?quiet:tone);
  if(r.started)started++;
  assert(!r.ended);
}
for(let i=0;i<40;i++){
  const r=vad.feed(quiet);
  if(r.ended&&r.chunks){ended++;total=r.chunks.reduce((n,x)=>n+x.length,0);}
}
assert.equal(started,1);assert.equal(ended,1);assert(total>rate*29);
const pcm=pcm16([new Float32Array([1,1,-1,-1])],48000);
const view=new DataView(pcm);assert.equal(pcm.byteLength,4);assert.equal(view.getInt16(0,true),32767);assert.equal(view.getInt16(2,true),-32768);
console.log('PASS: long-turn silence boundaries and PCM resampling');
