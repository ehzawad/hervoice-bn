# HerVoice BN — a Bengali speech-to-speech voice bot for the browser

Speak Bengali into a web page; it transcribes, thinks, and speaks Bengali back, interrupting
itself when you start talking again. Four processes on one 24 GB GPU, no cloud services, no
vendor APIs. Every model is local.

```
browser ──WebSocket──▶ gateway ──HTTP──▶ asr   FastConformer-CTC   ehzawad/stt_bn_fastconformer_ctc
         (16 kHz PCM)  Silero VAD  │      llm   Gemma 4 E4B         google/gemma-4-E4B-it-qat-w4a16-ct (vLLM)
        ◀─24 kHz PCM─  turn loop   └──▶  tts   IndicF5 + Vocos     ehzawad/indicf5-bangla-tts
```

## Measured

On one RTX A5000, warm medians. The scripts that produced every number are in `hervoice/eval/`.

| | |
|---|---|
| **Time to first audio, from when you stop speaking** | **3.07 s** median, 3.49 s p90 (30 real speakers) |
| …of which the end-of-turn silence wait | 700 ms |
| …of which TTS | ~1.6 s — IndicF5 is flow matching, so a whole sentence must finish before any sample exists |
| ASR 68 ms · brain first token 34 ms | the cheap parts |
| Barge-in, cancel → new turn | 631 ms, zero stale audio frames |
| ASR on 30 real spontaneous Bengali speakers | CER median 0.074, mean 0.115, p90 0.235 |
| Valid Bengali replies on those 30 | 30/30, no utterance cut short |
| Conversation memory, follow-ups needing earlier turns | 10/11 through audio; 10/11 in text |
| Resident VRAM, all four services | 19.1 GiB |

The first-audio figure is measured from the **end of the user's speech**, which includes the
silence the bot waits through before it even knows you stopped. Measured from the endpoint
instead it is 2.37 s, and that is the number most systems quote; it is not what a person
experiences, so this repo quotes the larger one.

`results/` holds the JSON behind every figure above.

## What is not proven

- **No human has spoken to it.** The box has no microphone. Everything is file-driven: scripted
  dialogues in a synthetic voice, and 30 real recordings from IndicVoices-R.
- **Echo cancellation is unverified.** The client requests `echoCancellation: {exact: "all"}`
  (a spec MUST for non-WebRTC playback on Chrome 141+) with a boolean fallback, but an accepted
  constraint is not evidence that a real speaker/microphone pair cancels the bot's own voice.
  If it fails, the bot barges in on itself.
- **NFE 16 has no quality comparison.** It halves TTS latency against 32; what it costs in
  pronunciation and naturalness has not been measured.
- **Real speakers are West Bengal**, not Bangladeshi, and are clean recordings, not phone or
  laptop microphones.

## Run it

Four processes, supervised by a script that verifies pid, owner and cwd before signalling
anything — never by matching command-line patterns.

```bash
uv venv --python 3.12 .venv-bnweb
uv pip install --python .venv-bnweb/bin/python torch==2.8.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-bnweb/bin/python -r requirements-bnweb.txt onnxruntime
git -C third_party clone https://github.com/SWivid/F5-TTS
git -C third_party/F5-TTS checkout 9c614e9657089213efc6a7421b30630be138a3f5
uv pip install --python .venv-bnweb/bin/python -e third_party/F5-TTS

uv venv --python 3.12 .venv-vllm && uv pip install --python .venv-vllm/bin/python vllm

export HV_GW_TOKEN=$(python3 -c "import secrets;print(secrets.token_urlsafe(32))")
hervoice/svc/run.sh start          # llm → asr → tts → gateway, each verified before the next
hervoice/svc/run.sh status
```

Then, from your own machine:

```bash
ssh -N -L 8100:127.0.0.1:8100 <box>
open http://localhost:8100/
```

The tunnel is not only for privacy: `getUserMedia` requires a secure context, and
`http://localhost` counts as one. On a plain `http://<ip>` page `navigator.mediaDevices` is
**undefined** and the microphone button cannot work at all. For a real deployment see
`deploy/` (Caddy terminates TLS and proxies the WebSocket).

## Test it without a microphone

```bash
hervoice/svc/smoke.py            # service path only: ASR → LLM → TTS latency
hervoice/svc/simulate_ws.py --scenario bargein    # barge-in and stale-audio drop, over a real socket
hervoice/eval/run_scenarios.py --mode text        # conversation memory, no audio
hervoice/eval/run_scenarios.py --mode audio       # the same dialogues through the gateway
hervoice/eval/run_real_speech.py --n 30           # 30 real spontaneous speakers
hervoice/eval/eval_turn_detect.py --n 30          # endpointing policy comparison
```

`hervoice/eval/scenarios_bn.json` holds six scripted Bengali dialogues where later turns are
unanswerable without earlier ones — pronouns, ellipsis, a name to recall, a topic dropped and
resumed. `render_user_voice.py` speaks the user's side in a real Bengali male voice, distinct
from the assistant's.

## How it works, where it is unusual

**Turn taking.** Silero VAD with a 700 ms end-of-turn silence. The default 220 ms fires at
natural mid-sentence pauses in spontaneous Bengali: on 30 real clips it cut off 15 of them, and
7.8 s of speech was transcribed as a single word. 700 ms cuts off 1 of 30. A semantic end-of-turn
model was evaluated and rejected — see `docs/DECISIONS.md`.

**Barge-in.** Every audio chunk is stamped with an `(epoch, seq)`. A cancel bumps the epoch, so
the browser discards everything it still holds for the old one; the server cannot "unsend" audio
that is already in a playback buffer. `SPEAKING` is a distinct state from `THINKING`, because a
turn is not over when generation finishes — it is over when the listener has heard it.

**Memory.** Text history only: in a cascade the brain never sees audio, so storing audio would
cost tokens and buy nothing. What makes this unusual is *when* it commits: a sentence enters the
conversation only once the client has **acknowledged playing every chunk of it**. Anything cut
off becomes an explicit interruption marker instead of text the assistant is told it said.

**Failure honesty.** Dropped input frames mark the turn degraded and say so rather than
silently producing a wrong transcript. A generation thread that outlives its join wedges the
session instead of letting a second turn start on the same GPU. Auth fails closed.

## Layout

```
hervoice/svc/     the service: gateway, turn loop, engine, conversation, the three services
hervoice/bn/      model construction (IndicF5 with its two pinned DiT defaults, ASR, brain)
hervoice/live/    turn_detector.py — Silero VAD hysteresis, from omni-voice-lab
hervoice/eval/    scenarios, real-speech robustness, turn-detector comparison, user-voice render
deploy/           compose.yaml, three Dockerfiles, Caddyfile — not yet built or run
docs/DECISIONS.md every decision and the evidence that settled it, including what was withdrawn
```

Licence MIT (`LICENSE`); third-party code and model terms in `THIRD_PARTY.md`. The IndicF5
terms require permission for voice cloning: the default assistant voice is ai4bharat's own
released reference prompt.
