import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const elements = new Map();
let stopped=0, ended=0;
const track={stop(){stopped++},onended(){},onmute(){}};
const box=vm.createContext({
 document:{getElementById(id){if(!elements.has(id))elements.set(id,{classList:{remove(){},add(){}},disabled:false});return elements.get(id)}},
 WebSocket:{OPEN:1},clearInterval(){},setTimeout,console
});
const src=fs.readFileSync('app/native_voice/ui/gemini.js','utf8');
vm.runInContext(src.slice(0,src.indexOf("$('start').onclick")),box);
box.track=track;box.sent=[];box.onClose=()=>ended++;
vm.runInContext("live=true;connected=true;stream={getTracks:()=>[track]};socket={readyState:1,send:x=>sent.push(x),close:onClose};context={state:'running',close:async()=>{}};token='bye';sources.add({});audioFinished=true;",box);
await vm.runInContext("receive({type:'call_closing'})",box);
assert.equal(stopped,1);
assert.equal(vm.runInContext('connected',box),false);
await vm.runInContext("receive({type:'call_end_after_playback'})",box);
assert.equal(vm.runInContext('live',box),true,'Must finish buffered goodbye');
vm.runInContext("sources.clear();maybePlayed()",box);
assert.equal(vm.runInContext('live',box),false);
assert.equal(ended,1);
assert(box.sent.some(x=>JSON.parse(x).type==='end'));
console.log('PASS: microphone stops before farewell playback; call closes after queued audio drains');
