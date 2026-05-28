# vlog-automation

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-GPU_accelerated-76B900?style=flat-square&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-363739?style=flat-square)
![Status](https://img.shields.io/badge/status-active-CCFF00?style=flat-square)
![Version](https://img.shields.io/badge/version-9.1-CCFF00?style=flat-square)

AI-powered video editor. Drop in raw OBS footage, get back a tightly cut video with subtitles.

Transcribes with Whisper, scores every speech segment, removes dead air, renders with GPU. 40+ hours of footage processed for 3 KL creators.

---

## How it works

```
raw .mp4
   │
   ▼
[Phase 1] Transcription  — faster-whisper large-v3, word-level timestamps, Malay+English
   │
   ▼
[Phase 2] Scoring        — transcript weight + audio energy + visual activity per segment
   │
   ▼
[Phase 3] Topic grouping — gaps > 3.5s = new topic; topics > 90s split automatically
   │
   ▼
[Phase 4] Selection      — top segments by quality mode (Highlights 62% / Balanced 38% / Chill 12%)
   │
   ▼
[Phase 5] Rendering      — ffmpeg concat + NVENC GPU encode
   │
   ▼
[Phase 6] Subtitles      — kinetic .ass + .srt (2-word UPPERCASE chunks, auto-burned)
   │
   ▼
edited .mp4 + subtitles
```

## Genre presets

Scoring weights differ by content type:

| Genre | Transcript | Audio | Visual |
|---|---|---|---|
| Discord Call | 55% | 35% | 10% |
| Gaming | 30% | 30% | 40% |
| Vlog | 45% | 35% | 20% |

## Quality modes

| Mode | Cut target | Use case |
|---|---|---|
| Highlights | 62% kept | Short-form, reels |
| Balanced | 38% kept | Standard upload |
| Chill | 12% kept | Light trim only |

## Stack

- [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-based Whisper, runs large-v3 on GPU
- `ffmpeg` — cutting, concatenation, NVENC encoding
- `CUDA` — GPU transcription + rendering (falls back to CPU)
- `tqdm` — progress tracking

## Requirements

```
Python 3.10+
ffmpeg  (winget install ffmpeg)
NVIDIA GPU with CUDA  (optional — CPU fallback available)
```

## Install

```bash
git clone https://github.com/arifaqyl/vlog-automation
cd vlog-automation
python -m venv venv
venv\Scripts\activate
pip install faster-whisper tqdm
```

## Usage

```bash
# Windows — double-click RUN.bat
# or run directly:
python auto_cutter.py
```

Prompts:
1. Mode — Auto-cut or Manual trim
2. Drop your `.mp4` path
3. Genre — Gaming / Discord / Vlog / Auto
4. Quality — Highlights / Balanced / Chill
5. Wait. Review in `review.html`. Get final `.mp4`.

## Reaction word detection

Whisper romanizes Malay, so `gila` stays `gila`. Detector covers:

```
English: haha, bruh, bro, damn, yo, wait, clutch, gg, rip, omg ...
Malay:   gila, babi, pergh, walao, sial, bodoh, mampus, harap, eh ...
```

Reaction density boosts segment score — squad banter, clutch moments, and highlights surface automatically.

## Telegram done notification (v9.1)

Add `notify.py` to the same folder and set env vars:

```bash
export TG_BOT_TOKEN="your_bot_token"
export TG_CHAT_ID="your_chat_id"
```

When a job finishes you get:

```
vlog-automation done
Output: output_20260528_143012.mp4
1:23:45 → 0:52:10  (62% kept, 0:31:35 cut)
Mode: DISCORD CALL / HIGHLIGHTS
```

Every run is also appended to `processing_log.json` for usage stats.

---

**[arifaqyl.me](https://arifaqyl.me)** · [github.com/arifaqyl](https://github.com/arifaqyl)
