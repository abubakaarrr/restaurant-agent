class Capture extends AudioWorkletProcessor {
  constructor(){super();this.buffer=new Float32Array(2048);this.offset=0;}
  process(inputs){const samples=inputs[0]?.[0];if(samples){for(let i=0;i<samples.length;i++){this.buffer[this.offset++]=samples[i];if(this.offset===this.buffer.length){this.port.postMessage(this.buffer);this.buffer=new Float32Array(2048);this.offset=0;}}}return true;}
}
registerProcessor('microphone-capture',Capture);
