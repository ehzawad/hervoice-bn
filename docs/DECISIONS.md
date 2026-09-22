# Decisions, and what decided them

Every entry names the evidence. Where the evidence is a measurement on this box, the script
that produced it is in `hervoice/eval/` or `hervoice/svc/` and the numbers are reproducible.
Where it is research, it says so, and what refuted or confirmed it.

## Architecture

**Four processes, three dependency sets, one gateway.** vLLM (torch 2.13, CUDA 13.0) serving
Gemma 4 E4B; a NeMo FastConformer-CTC ASR service and an IndicF5 TTS service (both torch 2.8,
CUDA 12.8 — `nemo_toolkit[asr]==2.7.3` needs `transformers>=4.57`, incompatible with the
research repo's 4.51 pin, so this stack cannot share a venv with it); a CPU-only gateway that
owns the WebSocket, the VAD and the turn state machine.

**Measured, not designed:** the microservice split came out *faster* than a single co-resident
process — 1752 ms vs 2818 ms to first audio — because vLLM's prefill (34 ms to first token)
beats HF `generate()` (430 ms) by more than the HTTP hops cost (~2 ms). `bench_coresident.py`
vs `smoke.py`.

**Gemma 4 E4B, not Qwen.** Qwen2.5-3B answered "capital of Bangladesh" with "কুব达রা" (with a
Chinese character); Qwen2.5-7B with "কার্তিক". Size did not fix Bengali. Gemma answers "ঢাকা".
Google's `-qat-w4a16-ct` checkpoint is described on its card as "for native, optimized
inference with vLLM"; it occupies ~16.9 GiB at utilisation 0.65.

**No Triton.** vLLM is already a serving layer; the barge-in state machine cannot be a Triton
ensemble; ASR and TTS are single-tenant. Triton would add a process and a protocol hop.

**No framework (Pipecat, LiveKit, ...).** Research (two independent reviews) and the council
agree: every framework exists to glue cloud vendors, and all three models here are local. The
gateway already does what Pipecat has an open bug about (#5305, unspoken text appended to
context) and what its browser-WebSocket path cannot do (stale-audio flush). If ever: Pipecat
(BSD-2), not LiveKit (proprietary model licence; 14 turn-detector languages, no Bengali). A
correction we accepted: vendors *do* list Bengali (Deepgram, Gemini Live, Azure); the
defensible claim is that none hosts *these* models or publishes a Bengali WER.

## Turn taking

**End-of-turn silence: 700 ms** (`HV_MIN_SILENCE_MS`). Silero's default 220 ms fires at natural
mid-sentence pauses in spontaneous speech: 7.8 s of speech was transcribed as one word, and
the speaker's continuation became a barge-in. Offline sweep on 30 real IndicVoices-R extempore
clips (`eval_turn_detect.py`), clips cut off / median added latency:

| 220 | 350 | 500 | 600 | 700 | 800 | 1000 |
|---|---|---|---|---|---|---|
| 15/30 · 242 ms | 9 · 368 | 6 · 528 | 3 · 624 | **1 · 722** | 1 · 817 | 0 · 1042 |

600→700 took cut-off clips from 3 to 1 for +98 ms; a cut-off destroys the referent *and* turns
the rest into a barge-in, so the exchange is worth it. Confirmed end-to-end at 600 ms: real-
speech ASR CER median 0.207→0.078, p90 0.878→0.345. Unproven for short questions; the
synthetic scripted questions were also cut at 220 ms, so "short questions are safe" cannot be
assumed.

**Smart Turn v3: evaluated, refuted, not adopted.** Research recommended it (BSD-2, Whisper-tiny
encoder, published Bengali accuracy 83.8 %). Measured on the same 30 clips with Pipecat's exact
preprocessing (last 8 s, left-padded, vendored Whisper log-mel, sigmoid > 0.5, 3 s fallback):
cut-off clips 15 → 13, real ends detected 25/30 (five misses), confidence 0.94–0.98 at mid-
sentence pauses, **101 ms per decision** on one CPU thread (marketed 12 ms). Caveat: the
clips are dataset segments, not human-labelled turn ends. Neither caveat rescues it.
`eval_turn_detect.py` and `whisper_features.py` stay so the result is reproducible.

## Memory

**Text history only.** In a cascade the brain sees text; audio history would cost 3–10× the
tokens and buy nothing. Research: cascades beat end-to-end on multi-turn benchmarks (URO-Bench
83.4 vs 68.4); production frameworks declare audio-content types and never construct them;
OpenAI drops audio tokens when a transcript exists. 12 messages / 2400 characters, no
summarisation in v1 (an extra LLM call with hallucination risk). Prefix-cache-aware trimming
was judged premature at this window size (brain first token is 34 ms).

**Commit from playback acknowledgement, not emission.** The engine's "spoken" meant "handed to
the socket"; the browser can hold several chunks and discard them on a cancel. Every chunk
carries `(epoch, seq)` assigned by the turn loop; the worklet acks each chunk as it finishes
rendering and reports what a flush dropped; a sentence enters history only when every chunk
is acked; otherwise an explicit Bengali interruption marker is stored, never invented speech.
Offline unit test: all-acked → 3/3 sentences; barge-in after one sentence → 1/3 + marker;
no acks + timeout → marker only. A rendered-sample ack is a delivery proxy, not proof a person
heard it.

## Latency

**Measured: 3.07 s median, 3.49 s p90, from when the user stops speaking** (30 real speakers;
`first_audio_from_speech_end_ms`). From the endpoint instead it is 2.37 s — that is the figure
most systems quote and the one quoted here earlier, and it understates the experience by the
whole silence threshold. The timer also stops at server emission, not at the listener's ear.
TTS is ~92 % of the service-path figure (IndicF5 is flow matching: the whole chunk must finish
before any sample exists).

**NFE sweep, measured on both cards** (`bench_nfe.py`, 8 assistant sentences, same seeds,
re-ASR CER through the bot's own Bengali ASR, ECAPA drift against the NFE 32 rendering):

| NFE | A5000 ms | A6000 ms | CER A5000 | CER A6000 | timbre vs 32 |
|---|---|---|---|---|---|
| 8 | 559 | 469 | **0.0429** | 0.0077 | 0.955 |
| 12 | 840 | 682 | 0.0045 | 0.0192 | 0.974 |
| 16 | 1133 | 911 | 0.0045 | 0.0128 | 0.984 |
| 24 | 1723 | 1372 | 0.0045 | 0.0128 | 0.985 |
| 32 | 2323 | 1824 | 0.0045 | 0.0128 | 1.000 |

**NFE 8 is not safe.** Identical sentences and seeds gave CER 0.0077 on one card and 0.0429
on the other — ten times its own reference. Below roughly 12 steps the ODE solve becomes
sensitive to floating-point differences, so a single good run proves nothing. NFE 12 is at the
reference CER on both cards and saves ~290 ms per sentence against 16; its timbre drift
(0.974 vs 0.984) is small but real and was put to a blind listening test rather than decided
by metric. Duration is identical at every setting because it comes from the byte-ratio
heuristic, not the solver — a duration ratio of 1.000 proves nothing about pacing.

**The A6000 is ~1.25× faster than the A5000 at every step count**, despite identical memory
bandwidth (768 GB/s). This synthesis is compute-bound, not bandwidth-bound — which is worth
remembering before assuming a cloud L4 (300 GB/s but comparable fp32 throughput) would be
proportionally slower.

Levers, ranked by evidence: NFE below 16 (above); a deliberately short first sentence; int16 on the wire (halves
bandwidth, not latency); smaller first chunk. Not levers: framework, WebRTC, another GPU type
— none has measured evidence here.

## Evaluation without a human

The box has no microphone. User turns for the scripted dialogues are synthesised with the
*released* IndicF5 conditioned on a real Bengali male speaker — a different voice from the
assistant's — and are clean synthetic audio. Robustness uses 30 real spontaneous West Bengal
speakers (IndicVoices-R), SNR-stratified; the dialect differs from the Bangladeshi target.
The LLM judge is Gemma grading itself and is reported as a weak signal beside keyword hits
and referent resolution. Known: one Russian word and one Korean fragment appeared in 30 real-
speech replies before the code-switch checker was widened.

**The harness was once the bottleneck, and it took an isolation test to see it.** The user
voice was originally IndicF5 on ai4bharat's *Marathi* prompt. Audio-mode memory scored 5–6/10
against text mode's 10/11 and the gap survived the endpoint fix. Sending the rendered turns
straight to the ASR service, bypassing the gateway, settled it: synthetic CER 0.282 direct
versus 0.261 through the gateway versus 0.112 for real humans. The gateway added nothing; the
*test voice* was harder to recognise than real Bengali speakers, because cross-lingual
prompting gives IndicF5 a non-Bengali accent ("ফি কত?" came back as "হ"). Re-conditioned on a
real Bengali speaker: CER 0.282 → 0.135, and memory 5–6/10 → 10/11. Isolate the harness before
blaming the system, and condition evaluation TTS on the target language.

## Deployment

Three images (`vllm/vllm-openai:v0.29.0` unchanged; one `bn-gpu` image with an `asr|tts`
entrypoint; a CPU-only gateway), Compose with health-gated ordering, Caddy for TLS and the
WebSocket upgrade, weights in a volume. Any host driver ≥ 580.65.06 serves both CUDA stacks
(this box runs 610.43.02 with both). vLLM gets an absolute `--kv-cache-memory` (the 2.89 GiB
the measured run used) so its footprint stops depending on startup order. `getUserMedia` is
undefined on plain http off localhost: there is no "open a port" path to a working microphone.

Cloud (research, unverified on hardware): GCP `g2-standard-8` (L4) in Mumbai ≈ $213/month
compute at 8 h/day; AWS `g6.2xlarge` or `g5.2xlarge` (A10G) as fallback. **Before choosing:
benchmark TTS on an L4 and an A10G** (~$3 of spot time) — the L4 has 39 % of the A5000's
memory bandwidth and no F5-TTS benchmark exists on either. T4 does not fit (12.69 GiB floor
before KV cache; no bf16). GCP's free trial cannot use GPUs or request GPU quota; AWS GPU
quota starts at 0 vCPUs. Bangladesh's Personal Data Protection Act (Act 63, 10 April 2026) is
extraterritorial and whether voice is "biometric" is unresolved: counsel before real users'
voice leaves the country.

## Things that were claimed and then withdrawn

- "1–2 s to first audio" — unmeasured when said; withdrawn; measured 1752 ms service-path,
  which itself excludes the silence wait.
- "Smart Turn v3 as the principled fix" — refuted by measurement (above).
- "29/30 clean Bengali replies" — a Korean fragment slipped the checker; at most 28/30.
- "3–4 conversations per card, second GPU at 5" — KV-cache arithmetic, not throughput; withdrawn.
- "memory is correct" — narrowed to "history use is demonstrated in text mode (10/11);
  audio-mode resolution was 5–6/10 and a second cause is under investigation."
