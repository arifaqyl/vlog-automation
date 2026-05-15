\# Vlog Automaton



AI-powered video editor that auto-cuts gaming sessions, Discord calls, and vlogs using speech transcription and audio analysis.



\## Overview



Drop in raw OBS footage, get back a tightly edited video with subtitles. Built for Malaysian content creators — detects Malay/English reactions, squad callouts, and gaming moments automatically.



\## Features



\- \*\*AI transcription\*\*: Whisper speech-to-text with word-level timestamps

\- \*\*Smart cutting\*\*: Removes dead air, filler words, boring stretches

\- \*\*Genre presets\*\*: Tuned separately for Gaming, Discord Calls, and Vlogs

\- \*\*Quality modes\*\*: Highlights (62% cut), Balanced (38% cut), Chill (12% cut)  

\- \*\*Malay/Manglish support\*\*: Detects "gila", "pergh", "walao", "sial" as reaction markers

\- \*\*Kinetic subtitles\*\*: Auto-generates styled .ass + .srt subtitle files

\- \*\*GPU accelerated\*\*: NVIDIA NVENC support for fast rendering

\- \*\*Review UI\*\*: HTML interface to adjust cuts before final render

\- \*\*Manual trim mode\*\*: Specify exact timestamps to keep



\## Requirements



Python 3.10+

ffmpeg (winget install ffmpeg)

CUDA-capable GPU (optional, falls back to CPU)



\## Installation



```bash

git clone https://github.com/arifaqyl/vlog-automaton

cd vlog-automaton

python -m venv venv

venv\\Scripts\\activate

pip install faster-whisper tqdm

```



\## Usage



```bash

\# Windows - double click RUN.bat

\# Or run directly:

python auto\_cutter.py

```



Then follow the prompts:

1\. Choose mode: Auto-cut or Manual trim

2\. Drag your video file in

3\. Select genre (Gaming / Discord / Vlog)

4\. Select quality (Highlights / Balanced / Chill)

5\. Wait for processing

6\. Review cuts in review.html

7\. Get final .mp4 + subtitles



\## How It Works



\*\*Phase 1 — Transcription\*\*: faster-whisper transcribes speech with word-level timestamps



\*\*Phase 2 — Scoring\*\*: Each segment scored by:

\- Reaction words detected (Malay + English)

\- Audio energy levels

\- Speech density

\- Squad name mentions



\*\*Phase 3 — Topic grouping\*\*: Nearby segments grouped into topics, split at natural gaps



\*\*Phase 4 — Selection\*\*: Top segments selected based on quality mode percentage



\*\*Phase 5 — Rendering\*\*: ffmpeg cuts and concatenates with GPU acceleration



\*\*Phase 6 — Subtitles\*\*: Kinetic subtitles burned in (2-word UPPERCASE chunks)



\## Genre Weights



| Genre | Transcript | Audio | Visual |

|-------|-----------|-------|--------|

| Discord Call | 55% | 35% | 10% |

| Gaming | 30% | 30% | 40% |

| Vlog | 45% | 35% | 20% |



\## License



MIT



\---



\*\*Arif Aqyl\*\* • \[GitHub](https://github.com/arifaqyl) • \[arifaqyl.me](https://arifaqyl.me)

