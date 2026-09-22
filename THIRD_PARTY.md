# Third-party code and models

| what | where | licence | note |
|---|---|---|---|
| `hervoice/live/turn_detector.py` | from ehzawad/omni-voice-lab | (same author) | Silero-VAD hysteresis; unchanged |
| `hervoice/svc/whisper_features.py` | pipecat-ai/pipecat `_whisper_features.py` | BSD-2-Clause, (c) 2024-2026 Daily | vendored byte-identical so Smart Turn v3 gets the features it was trained on; used only by the evaluation |
| `hervoice/bn/f5_common.py`, `f5_text.py` | ehzawad/indicf5-bangla-tts | MIT (same author) | IndicF5 construction with the two pinned DiT defaults |
| Silero VAD | `silero-vad` (pip) | MIT | |
| Smart Turn v3 | pipecat-ai/smart-turn-v3 (Hub) | BSD-2-Clause | downloaded at eval time; NOT used by the service |
| IndicF5 base | ai4bharat/IndicF5 | MIT; terms require permission for voice cloning | reference prompt `prompts/PAN_F_HAPPY_00001.wav` is the default assistant voice |
| Gemma 4 E4B | google/gemma-4-E4B-it-qat-w4a16-ct | Gemma Terms of Use | served by vLLM |
| FastConformer-CTC Bengali | ehzawad/stt_bn_fastconformer_ctc | CC-BY-SA-4.0 | |
| IndicF5 Bengali fine-tune | ehzawad/indicf5-bangla-tts | MIT | |
| IndicVoices-R Bengali | ai4bharat/indicvoices_r | CC-BY-4.0 | evaluation only, read from the Hub cache; not redistributed |
| `hervoice/eval/audio/*.wav` | generated here with released IndicF5 + its Marathi prompt | -- | synthetic USER turns for the scripted scenarios; not human speech |
