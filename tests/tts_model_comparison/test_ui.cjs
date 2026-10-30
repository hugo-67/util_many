/* Regression smoke test for root-level tts_model_comparison.html.
 * Run: node tests/tts_model_comparison/test_ui.cjs
 * No npm packages, model downloads, or browser required.
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(path.resolve(__dirname, '../../tts_model_comparison.html'), 'utf8');
const code = html.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
assert.ok(code && code.length > 10000, 'Missing inline JS');
assert.match(html, /Programmer Reference Summary/);
assert.match(html, /<meta name="description"/);
assert.match(html, /<meta name="keywords"/);
for (const name of ['browser','kitten','pocket7','pocket18','kokoro','piper','espeak']) {
  assert.ok(html.includes('value="' + name + '"'), 'Missing model: ' + name);
}
const elements = new Map(), spoken = [], requested = [], workers = [];
class El {
  constructor(id) { Object.assign(this, {id, value:'', textContent:'', className:'',
    disabled:false, hidden:false, options:[], style:{}, handlers:{}, children:[], scrollHeight:0, scrollTop:0}); }
  addEventListener(type, cb) { (this.handlers[type] ||= []).push(cb); }
  fire(type) { for (const cb of this.handlers[type] || []) cb({target:this}); }
  replaceChildren(...children) { this.children = children; this.options = children; }
  append(...children) { this.children.push(...children); this.options.push(...children); }
  removeAttribute(name) { this[name] = ''; }
}
const el = id => elements.get(id) || (elements.set(id, new El(id)), elements.get(id));
el('text').value = 'A local TTS smoke test'; el('speed').value = '1';
const speech = {getVoices:()=>[{voiceURI:'en',name:'English',lang:'en-US',localService:true},{voiceURI:'uk',name:'British',lang:'en-GB',localService:true}],
  cancel(){},pause(){},resume(){},addEventListener(){},
  speak(u){spoken.push(u.text);setTimeout(()=>u.onend?.(),1)}};
class Audio {
  constructor(){this.duration=1;} pause(){} load(){} removeAttribute(){}
  play(){setTimeout(()=>this.onended?.(),1);return Promise.resolve();}
}
class AudioContext {
  constructor(){this.state='running';this.currentTime=0;this.destination={};}
  resume(){return Promise.resolve()} suspend(){return Promise.resolve()} close(){return Promise.resolve()}
  createBuffer(c,n,r){return {duration:n/r,copyToChannel(){}}}
  createBufferSource(){return {playbackRate:{value:1},connect(){},
    start(){setTimeout(()=>this.onended?.(),1)},stop(){}}}
}
const mem=new Map();
class Worker {
 constructor(url,options){this.url=url;this.options=options;workers.push(this)}
 postMessage(message){if(message.type==='load')setTimeout(()=>this.onmessage?.({data:{type:'error',error:'Pocket mock model unavailable'}}),1)}
 terminate(){}
}
const scope = {Worker,document:{getElementById:el,head:{append(script){setTimeout(()=>script.onerror?.(),1)}},
  body:{append(){}},createElement:id=>new El(id)},
  window:{AudioContext,speechSynthesis:speech,addEventListener(){}},
  Audio,AudioContext,SpeechSynthesisUtterance:class{constructor(text){this.text=text}},
  speechSynthesis:speech,navigator:{platform:'test',hardwareConcurrency:2,deviceMemory:4,
    storage:{estimate:async()=>({usage:0})}},crossOriginIsolated:false,
  localStorage:{getItem:k=>mem.get(k),setItem:(k,v)=>mem.set(k,v)},
  fetch:async u=>{requested.push(u);throw Error('No network in smoke test')},
  Blob,Float32Array,Uint8Array,ArrayBuffer,DataView,Set,Map,
  URL:class{static createObjectURL(){return 'blob:smoke'}static revokeObjectURL(){}
    constructor(u){this.href=u;this.protocol='https:';this.pathname='/7m/';this.username='';this.password='';}},
  Option:class{constructor(t,v){this.textContent=t;this.value=v}},
  performance:{now:()=>Date.now()},setTimeout,clearTimeout,console:{info(){},warn(){}}};
vm.runInNewContext(code,scope,{filename:'tts_model_comparison.js'});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{
  await sleep(5);assert.equal(requested.length,0,'Unexpected model download at page load');
  el('engine').value='browser';el('engine').fire('change');
  await sleep(25);
  assert.ok(spoken.some(t=>t.includes('Browser native TTS')),'No model-name announcement');
  assert.equal(requested.length,0,'Browser baseline must not fetch weights');
  el('voice').value='uk';el('voice').fire('change');await sleep(20);
  assert.ok(spoken.some(t=>t.includes('British')),'Voice change must automatically speak chosen voice');
  el('play').onclick();await sleep(15);
  assert.ok(spoken.some(t=>t.includes('A local TTS smoke test')),'Play did not speak editable text');
  el('engine').value='pocket7';el('engine').fire('change');await sleep(15);
  assert.match(el('status').textContent,/no confirmed compatible/i,
    'Missing 7M export must be reported, never disguised as 18M');
  assert.equal(requested.length,0,'Unconfigured 7M downloaded assets');
  assert.equal(el('play').disabled,true,'Unavailable engine must disable Play');
  el('engine').value='pocket18';el('engine').fire('change');await sleep(25);
  assert.equal(requested.length,0,'Pocket worker must be same-origin, without remote worker-script downloads');
  assert.equal(workers.length,1,'Pocket worker created only after selection');
  assert.equal(workers[0].url,'./tts_pocket18_worker.js','Pocket worker must be same-origin');
  assert.match(el('status').textContent,/Pocket mock model unavailable/i,
    'Network failures must be surfaced instead of silently switching engines');
  assert.equal(el('play').disabled,true);
  console.log('PASS standalone TTS page: seven choices, lazy load, backend and voice announcements, same-origin Pocket worker, 7M honesty');
})().catch(e=>{console.error(e);process.exitCode=1});
