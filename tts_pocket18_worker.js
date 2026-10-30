// Based on runvnc/tablefolk app/tts/pocket18-worker.js at commit 4dc2b5d881852a2c03aff025098e5cb6c833e2bf.
// MIT License; Copyright (c) 2026 Jason.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// Pocket TTS, the small English model (english_18m_refl_2026-10: 18.2M
// parameters, Kyutai, CC-BY-4.0), ONNX export by maxime/pocket-tts-18m-onnx.
// A second engine beside inference-worker.js (the 100M model), with the same
// messages, so pocket-tts-manager.js can run either:
//   in:  load, set_audio_port (a port to the PCM worklet), set_voice,
//        encode_voice {voiceId, audio}, delete_voice, generate {text, voice}, stop
//   out: status, runtime_info, voices_loaded, bundle_loaded, loaded,
//        voice_set, voice_encoded, generation_started, audio_chunk(_meta),
//        generation_metrics, stream_ended, error
//
// About 36 MB (int8 flow LM, the rest fp32) plus one preset voice's state
// (~4 MB, fetched alone with a Range request from voices.bin). The voice
// encoder (39 MB) loads only when a custom voice first speaks. Files are kept
// in Cache Storage. Runs on the CPU (WASM): the GPU belongs to the chat model.
//
// The format (bundle.json): the flow LM keeps a KV cache per layer
// (state_cache_<i>, [2, 1, L, 8, 64], L dynamic: the voice prompt, the text
// and the frames to generate) and a state_offset; the sampler head is
// "drifting" (one call maps noise to the next latent); the Mimi decoder
// keeps its own states, zeroed at the start of an utterance.

let BASE = "https://huggingface.co/maxime/pocket-tts-18m-onnx/resolve/main/english_18m_refl_2026-10/";
const ORT_VERSION = "1.20.0";
const SENTENCEPIECE_URL = "https://huggingface.co/spaces/KevinAHM/pocket-tts-web/resolve/main/sentencepiece.js";
const CACHE = "tablefolk-tts-pocket18";
const MAX_TOKENS_PER_CHUNK = 50, FIRST_DECODE = 4, DECODE = 6, CACHE_STEP = 64;

let ort = null, bundle = null, sp = null, bos = null, audioPort = null;
let textSession = null, mainSession = null, flowSession = null, decoderSession = null, encoderSession = null;
let ready = false, generating = false, currentVoice = "alba";
const presetStates = new Map();      // name -> voice state {tensors: {name: {dtype, shape, data}}}
const customAudio = new Map();       // id -> Float32Array (24 kHz), encoded on first use
const customStates = new Map();      // id -> voice state

const status = (text, state = "loading") => postMessage({ type: "status", status: text, state });

// ---- files (Cache Storage, so a second visit downloads nothing) ----
async function bytes(name, { range = null, key = BASE + name } = {}) {
    let cache = null;
    try { cache = await caches.open(CACHE); const hit = await cache.match(key); if (hit) return await hit.arrayBuffer(); } catch {}
    const res = await fetch(BASE + name, range ? { headers: { Range: `bytes=${range[0]}-${range[1] - 1}` } } : {});
    if (!res.ok || (range && res.status !== 206)) throw new Error(`${name}: HTTP ${res.status}`);
    const buf = await res.arrayBuffer();
    try { await cache?.put(key, new Response(buf.slice(0))); } catch {}
    return buf;
}
async function session(name, opts = {}) {
    return ort.InferenceSession.create(new Uint8Array(await bytes(name)), { executionProviders: ["wasm"], graphOptimizationLevel: "all", ...opts });
}

// ---- tensors ----
function zeros(dtype, shape) {
    const n = shape.reduce((a, b) => a * b, 1);
    return new ort.Tensor(dtype, dtype === "int64" ? new BigInt64Array(n) : dtype === "bool" ? new Uint8Array(n) : new Float32Array(n), shape);
}
const dynAxis = spec => { const a = Object.keys(spec.dynamic || {}); return a.length === 1 ? +a[0] : -1; };
function freshStates(specs, dynamicSize = 0) {
    const out = {};
    for (const s of specs) out[s.name] = zeros(s.dtype, s.shape.map(d => d === -1 ? dynamicSize : d));
    return out;
}
// A voice state's caches, copied into zeroed caches `capacity` long (empty
// slots must be zeros: the graph masks the tail).
function flowStateFrom(voice, capacity) {
    const out = {};
    for (const spec of bundle.graphs.flow_lm_main.states) {
        const t = voice.tensors[spec.name]; if (!t) throw new Error(`voice state has no ${spec.name}`);
        const axis = dynAxis(spec);
        if (axis < 0) { out[spec.name] = new ort.Tensor(spec.dtype, t.data.slice(), t.shape); continue; }
        const shape = t.shape.slice(); shape[axis] = capacity;
        const outer = t.shape.slice(0, axis).reduce((a, b) => a * b, 1), inner = t.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
        const from = t.shape[axis] * inner, to = capacity * inner, data = new Float32Array(outer * to);
        for (let i = 0; i < outer; i++) data.set(t.data.subarray(i * from, (i + 1) * from), i * to);
        out[spec.name] = new ort.Tensor(spec.dtype, data, shape);
    }
    return out;
}
function takeStates(state, result, specs) { for (const s of specs) state[s.name] = result[s.output]; }

// ---- voices ----
// PTVB1 records: name, then tensors (path, dtype 0 f32 / 1 i64, shape, bytes).
function parseVoice(buf, offset = 0) {
    const v = new DataView(buf), td = new TextDecoder(); let o = offset;
    const u16 = () => { const x = v.getUint16(o, true); o += 2; return x; }, u32 = () => { const x = v.getUint32(o, true); o += 4; return x; };
    const str = n => { const s = td.decode(new Uint8Array(buf, o, n)); o += n; return s; };
    const name = str(u16()), count = u16(), tensors = {};
    for (let i = 0; i < count; i++) {
        const path = str(u16()), code = v.getUint8(o++), rank = v.getUint8(o++), shape = [];
        for (let r = 0; r < rank; r++) shape.push(u32());
        const len = u32(), raw = buf.slice(o, o + len); o += len;
        tensors[path] = { dtype: code === 1 ? "int64" : "float32", shape, data: code === 1 ? new BigInt64Array(raw) : new Float32Array(raw) };
    }
    return { name, tensors, end: o };
}
async function presetVoice(name) {
    if (presetStates.has(name)) return presetStates.get(name);
    const range = bundle.voices?.[name]; if (!range) throw new Error(`no voice ${name}`);
    status(`Loading voice (${name})...`);
    let voice;
    try { voice = parseVoice(await bytes("voices.bin", { range: [range.start, range.end], key: BASE + "voices.bin?voice=" + name })); }
    catch (e) {
        // no Range support: the whole file (34 MB), then this voice from it
        console.warn("[tts18] range fetch failed, reading all of voices.bin", e);
        voice = parseVoice(await bytes("voices.bin"), range.start);
    }
    presetStates.set(name, voice);
    return voice;
}
function parseNpy(buf) {
    const v = new DataView(buf), major = v.getUint8(6), hl = major === 1 ? v.getUint16(8, true) : v.getUint32(8, true), ho = major === 1 ? 10 : 12;
    const head = new TextDecoder().decode(new Uint8Array(buf, ho, hl)), shape = head.match(/\(\s*([0-9,\s]+)\)/)[1].split(",").map(s => s.trim()).filter(Boolean).map(Number);
    return { data: new Float32Array(buf.slice(ho + hl)), shape };
}
// A custom voice: encode the sample, then run it through the flow LM as the
// prompt and keep the resulting state (as preset voices ship it).
async function customVoice(id) {
    if (customStates.has(id)) return customStates.get(id);
    const audio = customAudio.get(id); if (!audio) throw new Error("No custom voice loaded.");
    status("Preparing custom voice...");
    encoderSession ||= await session("mimi_encoder.onnx");
    if (bundle.insert_bos_before_voice && !bos) bos = parseNpy(await bytes("bos_before_voice.npy"));
    const n = audio.length - audio.length % bundle.frame_size || bundle.frame_size, pcm = new Float32Array(n); pcm.set(audio.subarray(0, n));
    const enc = (await encoderSession.run({ audio: new ort.Tensor("float32", pcm, [1, 1, n]) })).conditioning;
    let data = enc.data, frames = enc.dims[1];
    if (bos) { const c = new Float32Array(bos.data.length + data.length); c.set(bos.data); c.set(data, bos.data.length); data = c; frames += bos.shape[1]; }
    const specs = bundle.graphs.flow_lm_main.states, state = freshStates(specs, frames);
    const r = await mainSession.run({ sequence: new ort.Tensor("float32", new Float32Array(0), [1, 0, bundle.dims.latent]), text_embeddings: new ort.Tensor("float32", data, [1, frames, bundle.dims.text_embedding]), ...state });
    const tensors = {};
    for (const s of specs) { const t = r[s.output]; tensors[s.name] = { dtype: t.type, shape: [...t.dims], data: t.data.slice() }; }
    const voice = { name: id, tensors };
    customStates.set(id, voice);
    // the page keeps it (IndexedDB), so the next visit skips the encoder and the prefill
    postMessage({ type: "voice_state", id, tensors });
    return voice;
}
const voiceFor = name => name.startsWith("custom:") ? customVoice(name.slice(7)) : presetVoice(name);

// ---- text ----
const ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
function words(n) {
    if (n < 20) return ONES[n];
    if (n < 100) return TENS[Math.floor(n / 10)] + (n % 10 ? "-" + ONES[n % 10] : "");
    if (n < 1000) return ONES[Math.floor(n / 100)] + " hundred" + (n % 100 ? " and " + words(n % 100) : "");
    if (n < 1e6) return words(Math.floor(n / 1000)) + " thousand" + (n % 1000 ? " " + words(n % 1000) : "");
    return String(n);
}
function prepare(text) {
    let t = String(text || "").normalize("NFKD").replace(/[̀-ͯ]/g, "")
        .replace(/[“”]/g, '"').replace(/[‘’]/g, "'").replace(/[—–]/g, ", ").replace(/…/g, "...")
        .replace(/\*+|_+|#+/g, "").replace(/\s+/g, " ").trim();
    t = t.replace(/(\d+)\s*gp\b/gi, "$1 gold").replace(/\b\d{1,6}\b/g, m => words(+m));
    if (!t) return "";
    if (/[A-Za-z0-9]$/.test(t)) t += ".";
    return t[0].toUpperCase() + t.slice(1);
}
// One context per utterance; long ones split at sentence ends near the limit.
function chunks(text) {
    const ids = sp.encodeIds(text);
    if (ids.length <= MAX_TOKENS_PER_CHUNK) return [text];
    const out = []; let cur = "";
    for (const s of text.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [text]) {
        const next = (cur + " " + s).trim();
        if (cur && sp.encodeIds(next).length > MAX_TOKENS_PER_CHUNK) { out.push(cur); cur = s.trim(); } else cur = next;
    }
    if (cur) out.push(cur);
    return out;
}

// ---- generation ----
function gauss(n, std) {
    const d = new Float32Array(n);
    for (let i = 0; i < n; i++) { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); d[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * std; }
    return d;
}
const maxFrames = tokens => Math.ceil((tokens / bundle.max_gen.tokens_per_second + bundle.max_gen.padding_seconds) * bundle.frame_rate);
const tick = () => new Promise(r => setTimeout(r, 0));

function sendAudio(pcm, metrics) {
    if (audioPort) { audioPort.postMessage({ type: "audio", data: pcm }, [pcm.buffer]); postMessage({ type: "audio_chunk_meta", metrics }); }
    else postMessage({ type: "audio_chunk", data: pcm, metrics }, [pcm.buffer]);
}

async function generate(text, voiceName) {
    generating = true;
    status("Generating...", "running");
    audioPort?.postMessage({ type: "reset" });
    postMessage({ type: "generation_started", data: { time: performance.now(), sourceSampleRate: bundle.sample_rate } });
    const t0 = performance.now(); let firstAudioAt = null, frames = 0, work = 0;
    try {
        const prepared = prepare(text); if (!prepared) throw new Error("No text to generate");
        const voice = await voiceFor(voiceName);
        const parts = chunks(prepared), ids = parts.map(p => sp.encodeIds(p));
        const voiceFrames = Number(voice.tensors.state_offset.data[0]);
        const capacity = Math.ceil((voiceFrames + Math.max(...ids.map(i => i.length + maxFrames(i.length)))) / CACHE_STEP) * CACHE_STEP;
        const flowSpecs = bundle.graphs.flow_lm_main.states, decSpecs = bundle.graphs.mimi_decoder.states, D = bundle.dims.latent;
        const emptySeq = new ort.Tensor("float32", new Float32Array(0), [1, 0, D]);
        const emptyText = new ort.Tensor("float32", new Float32Array(0), [1, 0, bundle.dims.text_embedding]);
        // the decoder runs on across text chunks (smoother joins)
        const dec = freshStates(decSpecs);
        let first = true;
        for (let c = 0; c < parts.length && generating; c++) {
            const flow = flowStateFrom(voice, capacity);
            const tok = new ort.Tensor("int64", BigInt64Array.from(ids[c].map(BigInt)), [1, ids[c].length]);
            let emb = (await textSession.run({ token_ids: tok })).embeddings;
            if (emb.dims.length === 2) emb = new ort.Tensor("float32", emb.data, [1, ...emb.dims]);
            takeStates(flow, await mainSession.run({ sequence: emptySeq, text_embeddings: emb, ...flow }), flowSpecs);
            const n = parts[c].split(/\s+/).length, rule = bundle.eos.frames_after_eos, after = n <= rule.max_short_words ? rule.short : rule.long;
            let seq = new ort.Tensor("float32", new Float32Array(D).fill(NaN), [1, 1, D]), eos = null;
            const latents = []; let decoded = 0;
            const decode = async (count, last) => {
                const lat = new Float32Array(count * D);
                for (let i = 0; i < count; i++) lat.set(latents[decoded + i], i * D);
                const s0 = performance.now();
                const r = await decoderSession.run({ latent: new ort.Tensor("float32", lat, [1, count, D]), ...dec });
                takeStates(dec, r, decSpecs);
                const ms = performance.now() - s0; work += ms; decoded += count;
                const pcm = new Float32Array(r.audio.data);
                firstAudioAt ??= performance.now();
                sendAudio(pcm, { chunkDuration: pcm.length / bundle.sample_rate, genTimeSec: ms / 1000, isFirst: first, isLast: last });
                first = false;
            };
            const limit = maxFrames(ids[c].length);
            for (let step = 0; step < limit && generating; step++) {
                if (step && step % 4 === 0) await tick();
                const s0 = performance.now();
                const r = await mainSession.run({ sequence: seq, text_embeddings: emptyText, ...flow });
                // EOS is ignored for the first frames, then `after` more are kept
                if (eos === null && step >= bundle.eos.min_frames && r.eos_logit.data[0] > bundle.eos.threshold) eos = step;
                if (eos !== null && step >= eos + after) break;
                const x = new ort.Tensor("float32", gauss(D, Math.sqrt(bundle.temperature)), [1, D]);
                const latent = new Float32Array((await flowSession.run({ c: r.conditioning, x })).latent.data);
                takeStates(flow, r, flowSpecs);
                latents.push(latent); frames++;
                seq = new ort.Tensor("float32", latent, [1, 1, D]);
                work += performance.now() - s0;
                if (latents.length - decoded >= (first ? FIRST_DECODE : DECODE)) await decode(first ? FIRST_DECODE : DECODE, false);
            }
            if (generating && decoded < latents.length) await decode(latents.length - decoded, c === parts.length - 1);
        }
        const audioSec = frames * bundle.frame_size / bundle.sample_rate, rtfx = work ? audioSec / (work / 1000) : 0;
        postMessage({ type: "generation_metrics", metrics: { rtfx, genTime: work / 1000, totalTime: (performance.now() - t0) / 1000, audioDuration: audioSec, firstAudioMs: firstAudioAt == null ? null : firstAudioAt - t0 } });
    } catch (err) {
        console.error("[tts18] generation", err);
        postMessage({ type: "error", error: String(err?.message || err) });
    } finally {
        if (generating) { audioPort?.postMessage({ type: "stream-ended" }); postMessage({ type: "stream_ended" }); status("Finished", "idle"); }
        generating = false;
    }
}

// Chunk base64 conversion to avoid exceeding argument limits on mobile.
function toBase64(buf) {
    const data=new Uint8Array(buf); let text="";
    for(let i=0;i<data.length;i+=0x4000)
        text+=String.fromCharCode(...data.subarray(i,i+0x4000));
    return btoa(text);
}
// ---- loading ----
async function load() {
    status("Loading ONNX Runtime...");
    const m = await import(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.min.mjs`);
    ort = m.default || m;
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
    ort.env.wasm.simd = true;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 2, 2) : 1;
    postMessage({ type: "runtime_info", data: { backend: "WASM CPU (18M)", isolated: self.crossOriginIsolated, threads: ort.env.wasm.numThreads, hardwareConcurrency: navigator.hardwareConcurrency || 0 } });
    status("Loading the small Pocket-TTS model...");
    bundle = JSON.parse(new TextDecoder().decode(await bytes("bundle.json")));
    if (bundle.flow?.type !== "drifting") throw new Error("unexpected flow head: " + bundle.flow?.type);
    [textSession, mainSession, flowSession, decoderSession] = await Promise.all([
        session("text_conditioner.onnx"), session("flow_lm_main_int8.onnx"), session("flow_lm_flow.onnx"), session("mimi_decoder.onnx"),
    ]);
    status("Loading tokenizer...");
    if (!self.__sp) {
        const js = await (await fetch(SENTENCEPIECE_URL)).text();
        self.__sp = await import(URL.createObjectURL(new Blob([js], { type: "application/javascript" })));
    }
    sp = new self.__sp.SentencePieceProcessor();
    await sp.loadFromB64StringModel(toBase64(await bytes("tokenizer.model")));
    await presetVoice(currentVoice);
    ready = true;
    const voices = bundle.predefined_voices || Object.keys(bundle.voices || {});
    postMessage({ type: "voices_loaded", voices, defaultVoice: currentVoice, language: "english" });
    postMessage({ type: "bundle_loaded", language: "english", sampleRate: bundle.sample_rate, initialLoad: true });
    status("Ready", "idle");
    postMessage({ type: "model_status", status: "ready", text: "Ready (english, 18M)" });
    postMessage({ type: "loaded" });
}

self.onmessage = async e => {
    const { type, data } = e.data || {};
    try {
        if (type === "load") { if (data?.modelBase) { const u = new URL(data.modelBase); if (u.protocol !== "https:" || u.username || u.password || !u.pathname.endsWith("/")) throw new Error("Invalid model base URL"); BASE = u.href; } return await load(); }
        if (type === "set_audio_port") { audioPort = e.ports?.[0] || null; return; }
        if (type === "stop") { generating = false; status("Stopped", "idle"); return; }
        if (type === "set_language") { postMessage({ type: "voices_loaded", voices: bundle?.predefined_voices || [], defaultVoice: currentVoice, language: "english" }); return; }
        if (!ready) return postMessage({ type: "error", error: "Models are not loaded yet." });
        if (type === "encode_voice") {
            // kept as audio; encoded when it first speaks (the encoder is 39 MB)
            const id = data.voiceId || "custom";
            customAudio.set(id, new Float32Array(data.audio)); customStates.delete(id);
            postMessage({ type: "voice_encoded", voiceName: "custom:" + id });
            return;
        }
        if (type === "load_voice_bin") {
            // a precomputed voice record (tools/build-voice-states.mjs, voices.bin's format)
            const v = parseVoice(data.buf); customAudio.set(data.voiceId, new Float32Array(1)); customStates.set(data.voiceId, { name: data.voiceId, tensors: v.tensors });
            postMessage({ type: "voice_state", id: data.voiceId, tensors: v.tensors });
            postMessage({ type: "voice_encoded", voiceName: "custom:" + data.voiceId });
            return;
        }
        if (type === "load_voice_state") {
            // a voice state computed on an earlier visit
            customAudio.set(data.voiceId, new Float32Array(1)); customStates.set(data.voiceId, { name: data.voiceId, tensors: data.tensors });
            postMessage({ type: "voice_encoded", voiceName: "custom:" + data.voiceId });
            return;
        }
        if (type === "delete_voice") { customAudio.delete(data.voiceId); customStates.delete(data.voiceId); return; }
        if (type === "set_voice") { await voiceFor(data.voiceName); currentVoice = data.voiceName; postMessage({ type: "voice_set", voiceName: currentVoice }); status("Ready", "idle"); return; }
        if (type === "generate") { if (!generating) await generate(data.text, data.voice || currentVoice); }
    } catch (err) {
        console.error("[tts18]", err);
        postMessage({ type: "error", error: String(err?.message || err) });
    }
};
