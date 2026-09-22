# Was a Moshi-style Bengali model built? No — and this is why

The owner asked for Moshi (full-duplex speech-to-speech, turn-taking decided inside the
network) reimplemented for Bengali. Two gates were measured before deciding, and a codex
council of three roles reviewed them. All three said do not build it. The reasoning, with the
numbers, is below, so the decision can be revisited if the inputs change.

## Gate 1: the data does not exist

Moshi's duplex behaviour comes from multi-stream training on two-party audio with **separate
channels per speaker** — the paper uses Fisher, roughly 2000 hours of English telephone calls.
The Bengali holding available here is 7.49 hours across 242 speakers (IndicVoices-R), and
**every clip is one person on one channel**. There is no Bengali Fisher. Synthetic two-party
dialogue can be generated with the bot's own TTS, but imposed pauses and overlaps cannot
demonstrate that a model learned natural conversational timing — which is the entire thing
Moshi is for.

## Gate 2: the codec

The first gate compared each Mimi reconstruction against *its own* source recording. That is
not how generated speech is scored, and the council was right to reject the extrapolation
built on it (one data point, and cosine similarities do not compose multiplicatively).

This is the decision-relevant test instead: take the voice the bot ships today, route it
through Mimi, and score both against the INDEPENDENT reference clip that IndicF5 is cloning —
the same protocol used for voice-generation evaluation.

8 assistant sentences, loudness-normalised, ECAPA cosine against the reference prompt:

| condition | SECS vs reference | cost of adding Mimi |
|---|---|---|
| IndicF5 as it ships today | **0.696** | — |
| + Mimi, 8 codebooks (Moshi's operating point) | **0.534** | **−0.163** |
| + Mimi, 16 codebooks | 0.621 | −0.075 |
| + Mimi, 32 codebooks | 0.655 | −0.041 |

This is a **lower bound on the damage**, not an estimate of the outcome: it assumes a
Moshi-style model predicts the acoustically correct tokens perfectly. Any generation error
comes on top. At Moshi's own 8-codebook configuration the voice loses 0.163 before the model
has made a single mistake.

Training to emit 16 or 32 codebooks recovers most of it, at 2× and 4× the audio tokens per
frame — 16 and 32 sequential depth-transformer steps per 80 ms frame instead of 8, against a
real-time budget that the local English Moshi demo has not yet been shown to meet when warm.

## Gate 3, raised by the council and not yet measured

The local English Moshi demo recorded **178 ms average per 80 ms frame**, but that figure
includes first-step CUDA-graph compilation. If it held when warm, the model would fall behind
real time by roughly 1.23 s per second of input. Sustained warm frame timing on this hardware
has not been established, and should be, before any Bengali adaptation is contemplated.

## Also established, and useful regardless

Mimi is **strongly sensitive to input loudness**: FLEURS English at its native level scored
0.304 speaker similarity at 8 codebooks and 0.671 after RMS normalisation — 0.37 from gain
alone. Before that was found, the same comparison appeared to show Mimi favouring Bengali over
English by +0.41. It does not: level-matched, Bengali 0.713 and English 0.671. **Mimi has no
Bengali penalty**; the 0.72 ceiling is simply what the codec does at 1.1 kbps in any language,
including the one Moshi itself speaks. Any Mimi pipeline must normalise loudness first, and a
streaming deployment needs a causal normaliser — whole-clip RMS uses information that is not
available at the start of a live utterance.

## What was done instead

The underlying want is turn-taking the model decides, rather than a VAD and a state machine.
The honest routes to a more conversational bot, ranked by measured evidence, are in
`DECISIONS.md`: NFE 12 (~290 ms, pending a listening verdict), and preparing ASR during the
silence window. A measured null result is recorded there too — prompting for a short first
sentence appears to save 480 ms but buys it entirely with filler; ban the filler and it is
worth 48 ms.

Moshi remains a separately gated research option, not a rejected idea. What would change the
verdict: a Bengali two-party separated-channel corpus appearing, or a decision to train at 16+
codebooks having first shown the real-time budget is met.
