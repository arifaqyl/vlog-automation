import os
import sys
import re
import glob
import subprocess
from collections import Counter
from tqdm import tqdm
from faster_whisper import WhisperModel

# ==========================================
# NVIDIA DLL FIX (Windows GPU)
# ==========================================
if sys.platform == "win32":
    venv_base = os.path.join(os.getcwd(), "venv", "Lib", "site-packages", "nvidia")
    for folder in ["cublas/bin", "cudnn/bin"]:
        p = os.path.join(venv_base, folder)
        if os.path.exists(p):
            os.add_dll_directory(p)
            os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]

# ==========================================
# CONFIG
# ==========================================
SPEECH_BUFFER = 0.45      # seconds padded around speech (anti word-clip)
TOPIC_GAP     = 3.5       # seconds gap = new topic boundary
MAX_TOPIC_SEC = 90        # split mega-topics longer than this
MERGE_GAP     = 1.2       # merge nearby intervals
MIN_CLIP_SEC  = 0.5       # drop micro-clips
VIDEO_EXTS    = {".mp4",".mkv",".avi",".mov",".webm",".flv",".wmv",".m4v"}

QUALITY = {
    1: {"name":"HIGHLIGHTS", "icon":"fire", "pct":62, "lo":0.22, "hi":0.48,
        "tag":"Fast-paced, only the bangers"},
    2: {"name":"BALANCED",   "icon":"zap",  "pct":38, "lo":0.42, "hi":0.72,
        "tag":"Entertaining, keeps context & flow"},
    3: {"name":"CHILL",      "icon":"wave", "pct":12, "lo":0.68, "hi":0.92,
        "tag":"Trims dead air & boring stretches"},
}

# Genre presets: tune scoring weights based on content type
# (transcript_w, audio_w, visual_w) must sum to 1.0
GENRE = {
    1: {"name": "DISCORD CALL",  "weights": (0.55, 0.35, 0.10),
        "tag": "Heavy banter, squad lore, voice-focused"},
    2: {"name": "GAMING",        "weights": (0.30, 0.30, 0.40),
        "tag": "Roblox, Valorant, Minecraft — visual action matters"},
    3: {"name": "VLOG",          "weights": (0.45, 0.35, 0.20),
        "tag": "IRL footage, talking head, mixed content"},
    4: {"name": "MIXED/AUTO",    "weights": (0.45, 0.35, 0.20),
        "tag": "Let the algorithm decide"},
}

# Reaction words (English + Malay/Manglish)
# Whisper romanizes everything, so "gila" stays "gila"
REACT = {
    # English reactions
    "haha","hahaha","hahahaha","lol","lmao","lmfao","rofl",
    "bruh","bro","dude","omg","nah","nope","damn","dang",
    "yo","yooo","wait","stop","dead","dying","killed",
    "gg","rip","clutch","goat","fire","lit","sick",
    "sus","cap","sheesh","ayo","what","noo","nooo",
    "holy","crazy","insane","why","boi","nice","lets","oh",
    "wow","howw","really","seriously","imagine","literally",
    # Malay/Manglish — exclamations
    "gila","sial","bodoh","bangang","wei","weh","woi","eh",
    "alamak","aduh","adoi","aiyo","aiyoh","aiyoyo",
    "pergh","fuyoh","fuyo","uish","ish","ishh",
    "walao","walawei","walaoeh","wahlao",
    "mampus","habis","hancur","pengsan","pengsan",
    "bestnya","dahsyat","power","terbaik","gempak",
    # Malay/Manglish — particles & fillers (high-density = natural speech)
    "lah","leh","lor","meh","kan","kot","je","aje","doh","duh",
    # Malay/Manglish — reactions & slang
    "takut","seram","comel","kelakar","lawak","gelak",
    "teruk","kesian","jahat","ganas","sabar","chill",
    "kenapa","apesal","apasal","macam","macamana","camne",
    "betul","betol","serious","serius","confirm",
    "jom","jomm","gooo","lesgo","lessgo",
    "mampos","mati","matilah","celaka","babi",
    "cantik","mantap","ohsem","awesome","terer",
    "abang","kakak","bang","kak",
    # Gaming slang (Malay gamers)
    "noob","pro","carry","feed","laggy","lag","toxic",
    "camper","tryhard","aimbot","hacker","cheater",
    # Whisper phonetic catches (how it hears Malay)
    "gilah","siall","weyy","oyy","haii","haish",
}

ENERGY_RE = [
    re.compile(r'!{2,}'),
    re.compile(r'\?{2,}'),
    re.compile(r'(.)\1{3,}'),
    re.compile(r'\b[A-Z]{3,}\b'),
]


# ==========================================
# SQUAD / LORE DICTIONARY
# ==========================================
# Your squad's names + phonetic variations (Whisper spells what it hears)
SQUAD_NAMES = {
    # Emy
    "emy", "emmy", "emi", "aimi",
    # Aqyl (Mohamad)
    "aqyl", "akil", "aqil", "mohamad", "mat",
    # Yan (Ayan)
    "yan", "ayan",
    # Ain Sufia Firhanah
    "ain", "sufia", "pia", "firhanah", "firhana",
    # Afini Pini
    "afini", "pini", "fini",
}


# ==========================================
# HELPERS
# ==========================================
def fmt(sec):
    """Seconds -> H:MM:SS or M:SS"""
    h, sec = int(sec // 3600), sec % 3600
    m, s = int(sec // 60), int(sec % 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def has_nvenc():
    """Check if NVIDIA GPU encoding is available."""
    r = subprocess.run('ffmpeg -hide_banner -encoders 2>&1',
                       shell=True, capture_output=True, text=True)
    return 'h264_nvenc' in (r.stdout + r.stderr)

def get_duration(path):
    r = subprocess.run(
        f'ffprobe -v error -show_entries format=duration '
        f'-of default=noprint_wrappers=1:nokey=1 "{path}"',
        shell=True, capture_output=True, text=True)
    try: return float(r.stdout.strip())
    except: return 0.0

def merge_intervals(ivs, gap=MERGE_GAP):
    if not ivs: return []
    ivs = sorted(ivs, key=lambda x: x[0])
    merged = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged

def _react_hits(words):
    return sum(1 for w in words if w.strip(".,!?\"'-:;()") in REACT)

def _energy_hits(text):
    return sum(len(p.findall(text)) for p in ENERGY_RE)

def _repetition(words):
    c = Counter(w for w in words if len(w) > 1)
    return sum(1 for _, n in c.items() if n >= 3)


# ==========================================
# MULTI-FILE: merge multiple videos into one
# ==========================================
def resolve_inputs(raw_path):
    """Accept single file, comma-separated files, quoted multi-drag, or a folder."""
    raw_path = raw_path.strip()
    paths = []

    # Check if it's a folder
    if os.path.isdir(raw_path.strip('"')):
        folder = raw_path.strip('"')
        for f in os.listdir(folder):
            ext = os.path.splitext(f)[1].lower()
            if ext in VIDEO_EXTS:
                paths.append(os.path.join(folder, f))
    else:
        # Split on multiple possible separators:
        # After get_inputs strips outer quotes, patterns look like:
        #   path1""path2  (from "path1""path2")
        #   "path1" "path2"  (space-separated with quotes)
        #   path1,path2  (comma-separated)
        if '""' in raw_path:
            # Most common: user drags two files, outer quotes stripped
            # e.g. D:\a.mp4""D:\b.mp4 → split on ""
            parts = [p.strip().strip('"') for p in raw_path.split('""') if p.strip()]
        elif '"' in raw_path and raw_path.count('"') >= 2:
            # "path1" "path2" style
            parts = re.findall(r'"([^"]+)"', raw_path)
            if not parts:
                parts = [raw_path.strip('"')]
        elif "," in raw_path:
            parts = [p.strip().strip('"') for p in raw_path.split(",")]
        else:
            parts = [raw_path.strip('"')]

        for p in parts:
            p = p.strip()
            if p and os.path.exists(p):
                paths.append(p)

    # Sort by creation time (chronological vlog order)
    paths.sort(key=lambda p: os.path.getctime(p))
    return paths


def merge_video_files(paths, work_dir):
    """Concatenate multiple videos into one temp file. Returns path."""
    if len(paths) == 1:
        return paths[0], False  # no merge needed

    print(f"\n  Merging {len(paths)} video files chronologically...")
    for i, p in enumerate(paths):
        print(f"    {i+1}. {os.path.basename(p)}  ({fmt(get_duration(p))})")

    list_path = os.path.join(work_dir, "_merge_list.txt")
    merged_path = os.path.join(work_dir, "_merged_input.mp4")

    with open(list_path, "w") as f:
        for p in paths:
            f.write(f"file '{os.path.abspath(p).replace(chr(92), '/')}'\n")

    # Try fast concat first (no re-encode)
    r = subprocess.run(
        f'ffmpeg -y -f concat -safe 0 -i "{list_path}" -c copy '
        f'-loglevel error "{merged_path}"', shell=True,
        capture_output=True, text=True)

    if r.returncode != 0 or not os.path.exists(merged_path):
        # Different formats — need re-encode
        print("  (Re-encoding to match formats...)")
        subprocess.run(
            f'ffmpeg -y -f concat -safe 0 -i "{list_path}" '
            f'-c:v libx264 -preset fast -c:a aac -loglevel error "{merged_path}"',
            shell=True)

    if os.path.exists(list_path):
        os.remove(list_path)

    print(f"  Merged: {fmt(get_duration(merged_path))}")
    return merged_path, True  # True = we created a temp file


# ==========================================
# AUDIO LOUDNESS (auto-editor inspired)
# ==========================================
def analyze_loudness(video_path):
    """Single-pass ebur128 loudness metering. Returns [(time, loudness_dB)]."""
    print("  Analyzing audio loudness (single pass)...")
    total_dur = get_duration(video_path)
    if total_dur <= 0:
        return []

    # ebur128 outputs momentary loudness (M:) every 0.1s to stderr
    cmd = f'ffmpeg -i "{video_path}" -af ebur128=peak=none -f null - -loglevel info'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    loudness = []
    for line in r.stderr.split('\n'):
        # Parse lines like: "[Parsed_ebur128_0 ...] t: 5.2    M: -23.1 ..."
        if 'M:' in line and 't:' in line:
            try:
                t_part = line.split('t:')[1].split('M:')[0].strip()
                m_part = line.split('M:')[1].split('S:')[0].strip()
                t_val = float(t_part)
                m_val = float(m_part)
                if m_val > -120:  # skip -inf silence markers
                    loudness.append((t_val, m_val))
            except (ValueError, IndexError):
                pass

    if loudness:
        # Downsample to ~1 sample per second for speed
        step = 1.0
        downsampled = []
        bucket_start = 0.0
        bucket_vals = []
        for t, v in loudness:
            if t >= bucket_start + step:
                if bucket_vals:
                    downsampled.append((bucket_start + step/2, max(bucket_vals)))
                bucket_start = t
                bucket_vals = [v]
            else:
                bucket_vals.append(v)
        if bucket_vals:
            downsampled.append((bucket_start + step/2, max(bucket_vals)))
        loudness = downsampled
        print(f"  {len(loudness)} loudness samples (1/sec)")
    else:
        print("  Audio analysis unavailable — using transcript only")
    return loudness


# ==========================================
# VISUAL MOTION SCORING (scene detection)
# ==========================================
def analyze_motion(video_path):
    """Detect scene changes via FFmpeg. Returns list of timestamps with high visual activity."""
    print("  Detecting visual activity (scene changes)...")
    total_dur = get_duration(video_path)
    if total_dur <= 0:
        return []

    # scene threshold 0.15 = detects moderate visual changes
    # showinfo outputs pts_time for each detected frame
    cmd = (
        f'ffmpeg -i "{video_path}" '
        f'-vf "select=\'gt(scene,0.15)\',showinfo" '
        f'-f null - -loglevel info'
    )
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    motion_times = []
    for line in r.stderr.split('\n'):
        if 'pts_time:' in line:
            try:
                pts = float(line.split('pts_time:')[1].split()[0])
                motion_times.append(pts)
            except (ValueError, IndexError):
                pass

    if motion_times:
        print(f"  {len(motion_times)} scene changes detected")
    else:
        print("  Visual analysis unavailable — using audio+transcript only")
    return motion_times


# ==========================================
# TOPIC DETECTION (never cut mid-conversation)
# ==========================================
def build_topics(segments):
    """
    Group consecutive segments into 'topics' based on gaps.
    A gap >= TOPIC_GAP seconds = new topic.
    Mega-topics (>MAX_TOPIC_SEC) get split at the largest internal gap
    so we don't score 8 minutes as one block.
    """
    if not segments:
        return []

    # Step 1: group by TOPIC_GAP
    raw_topics = []
    current = [segments[0]]
    for seg in segments[1:]:
        gap = seg["start"] - current[-1]["end"]
        if gap >= TOPIC_GAP:
            raw_topics.append(current)
            current = [seg]
        else:
            current.append(seg)
    if current:
        raw_topics.append(current)

    # Step 2: split mega-topics at largest internal gap
    topics = []
    for topic in raw_topics:
        topics.extend(_split_if_too_long(topic))

    return topics


def _split_if_too_long(topic_segs):
    """Recursively split a topic if it's longer than MAX_TOPIC_SEC."""
    if len(topic_segs) < 2:
        return [topic_segs]

    dur = topic_segs[-1]["end"] - topic_segs[0]["start"]
    if dur <= MAX_TOPIC_SEC:
        return [topic_segs]

    # Find the largest gap inside this topic
    best_gap = 0
    best_idx = len(topic_segs) // 2  # fallback: split in half
    for i in range(1, len(topic_segs)):
        gap = topic_segs[i]["start"] - topic_segs[i-1]["end"]
        if gap > best_gap:
            best_gap = gap
            best_idx = i
    left = topic_segs[:best_idx]
    right = topic_segs[best_idx:]

    # Recursively split each half if still too long
    return _split_if_too_long(left) + _split_if_too_long(right)


def score_topic(topic_segs, loudness_data, rms_min, rms_range, motion_data=None, genre_weights=None):
    """Score an entire topic (group of segments). Returns 0-100."""
    if not topic_segs:
        return 0

    all_text = " ".join(s["text"] for s in topic_segs)
    words = all_text.lower().split()
    wc = len(words)
    n = len(topic_segs)
    total_speech = sum(s["end"] - s["start"] for s in topic_segs)
    topic_span = topic_segs[-1]["end"] - topic_segs[0]["start"]
    if topic_span <= 0:
        topic_span = 0.1

    # --- Transcript signals ---
    avg_seg_dur = total_speech / n if n else 10

    # Density: how much of the topic span is actual speech
    density = min(total_speech / topic_span, 1.0)

    # Pace: segments per 30 seconds (normalized)
    pace = min((n / topic_span) * 30 / 12.0, 1.0)

    # Banter: rapid short exchanges (Discord energy)
    banter = 0.0
    if avg_seg_dur < 3.5 and n >= 3:
        banter = min((n / max(topic_span/30*8, 1)) * max(0, 3.5 - avg_seg_dur) / 3.5, 1.0)

    # Humor: reaction word density
    humor = min(_react_hits(words) / max(wc * 0.12, 1), 1.0) if wc else 0.0

    # Energy: caps, exclamation, repeated chars
    energy = min(_energy_hits(all_text) * 0.15, 1.0)

    # Word rate: words per second of speech
    word_rate = min((wc / total_speech) / 4.0, 1.0) if total_speech > 0 else 0.0

    # Repetition excitement
    rep = min(_repetition(words) * 0.2, 0.5)

    # --- Squad lore detection ---
    cleaned = [w.strip(".,!?\"'-:;()") for w in words]
    name_hits = sum(1 for w in cleaned if w in SQUAD_NAMES)
    lore_bonus = min((name_hits / max(wc, 1)) * 100, 25.0)

    # --- Chaos / overlap multiplier ---
    # Rapid-fire segments (< 0.8s gap) = everyone talking at once = hype
    rapid_count = 0
    for i in range(1, n):
        gap = topic_segs[i]["start"] - topic_segs[i-1]["end"]
        if gap < 0.8:
            rapid_count += 1
    chaos_mult = 1.0
    if rapid_count >= 4:
        chaos_mult = 1.3  # 30% boost for chaotic segments
    elif rapid_count >= 2:
        chaos_mult = 1.15  # 15% boost for moderately rapid

    # Multi-friend banter: 3+ different squad names = group chaos, never cut
    unique_names = len({w for w in cleaned if w in SQUAD_NAMES})
    if unique_names >= 3:
        chaos_mult *= 1.35  # 35% boost — group banter is gold
    elif unique_names == 2:
        chaos_mult *= 1.15  # 15% boost — 1-on-1 interaction

    # --- Audio loudness signal ---
    audio_score = 0.0
    if loudness_data and rms_range > 0:
        t_start = topic_segs[0]["start"]
        t_end = topic_segs[-1]["end"]
        samples = [rms for t, rms in loudness_data if t_start <= t < t_end]
        if samples:
            # Use both average and peak for better hype detection
            avg_rms = sum(samples) / len(samples)
            peak_rms = max(samples)
            # Blend average (70%) and peak (30%) — peak catches "WHAT?!" moments
            blended = avg_rms * 0.7 + peak_rms * 0.3
            audio_score = max(0, min(((blended - rms_min) / rms_range), 1.0))

    # --- Visual motion signal ---
    motion_score = 0.0
    if motion_data:
        t_start = topic_segs[0]["start"]
        t_end = topic_segs[-1]["end"]
        # Count scene changes within this topic's time window
        changes = sum(1 for t in motion_data if t_start <= t <= t_end)
        # Normalize: ~1 change per 5 seconds = high motion (0.2/sec)
        motion_rate = changes / max(topic_span, 0.1)
        motion_score = min(motion_rate / 0.2, 1.0)

    # --- Combined score (genre-tuned weights) ---
    transcript_score = (
        density    * 10 +
        pace       * 16 +
        banter     * 27 +
        humor      * 22 +
        energy     * 13 +
        word_rate  * 12 +
        rep        * 10 +
        lore_bonus       # squad name mentions
    )

    # Use genre weights if provided, otherwise smart defaults
    tw, aw, vw = genre_weights or (0.45, 0.35, 0.20)
    has_audio = len(loudness_data) > 0
    has_motion = motion_data is not None and len(motion_data) > 0

    if has_audio and has_motion:
        raw = transcript_score * tw + audio_score * 100 * aw + motion_score * 100 * vw
    elif has_audio:
        raw = transcript_score * (tw + vw) + audio_score * 100 * aw
    elif has_motion:
        raw = transcript_score * (tw + aw) + motion_score * 100 * vw
    else:
        raw = transcript_score

    return min(raw * chaos_mult, 100)


# ==========================================
# SELECTION (topic-level, never mid-conversation)
def select_topics(topics, topic_scores, total_dur, quality):
    """Pick which topics to keep based on scores + quality preset."""
    preset = QUALITY[quality]
    if not topic_scores:
        return []

    scores_sorted = sorted(topic_scores)
    idx = int(len(scores_sorted) * preset["pct"] / 100)
    threshold = scores_sorted[min(idx, len(scores_sorted) - 1)]

    # Select topics above threshold
    kept = []
    for topic, score in zip(topics, topic_scores):
        if score >= threshold:
            kept.append(topic)

    # Build intervals from kept topics (with buffer)
    intervals = []
    for topic in kept:
        t_start = max(0, topic[0]["start"] - SPEECH_BUFFER)
        t_end = topic[-1]["end"] + SPEECH_BUFFER
        intervals.append((t_start, t_end))

    merged = merge_intervals(intervals)
    merged = [iv for iv in merged if (iv[1] - iv[0]) >= MIN_CLIP_SEC]

    # Safety rails
    kept_dur = sum(e - s for s, e in merged)
    ratio = kept_dur / total_dur if total_dur else 0

    if ratio < preset["lo"]:
        # Not enough — lower threshold, keep more topics
        return _expand_topics(topics, topic_scores, total_dur, preset["lo"])
    if ratio > preset["hi"]:
        # Too much — raise threshold, keep fewer topics
        return _shrink_topics(topics, topic_scores, total_dur, preset["hi"])

    return merged


def _expand_topics(topics, scores, total_dur, target_ratio):
    """Keep topics by score until we hit target duration."""
    target = total_dur * target_ratio
    ranked = sorted(zip(scores, topics), key=lambda x: x[0], reverse=True)
    kept = []
    cum = 0
    # First pass: take best topics
    for sc, topic in ranked:
        dur = topic[-1]["end"] - topic[0]["start"]
        kept.append(topic)
        cum += dur
        if cum >= target:
            break
    # If still not enough, take all
    if cum < target:
        kept = [t for _, t in ranked]

    intervals = []
    for topic in kept:
        intervals.append((max(0, topic[0]["start"] - SPEECH_BUFFER),
                          topic[-1]["end"] + SPEECH_BUFFER))
    merged = merge_intervals(intervals)
    return [iv for iv in merged if (iv[1] - iv[0]) >= MIN_CLIP_SEC]


def _shrink_topics(topics, scores, total_dur, target_ratio):
    """Keep only top-scoring topics until we hit target duration."""
    target = total_dur * target_ratio
    ranked = sorted(zip(scores, topics), key=lambda x: x[0], reverse=True)
    kept = []
    cum = 0
    for sc, topic in ranked:
        dur = topic[-1]["end"] - topic[0]["start"]
        if cum + dur > target * 1.1:
            break
        kept.append(topic)
        cum += dur

    intervals = []
    for topic in kept:
        intervals.append((max(0, topic[0]["start"] - SPEECH_BUFFER),
                          topic[-1]["end"] + SPEECH_BUFFER))
    merged = merge_intervals(intervals)
    return [iv for iv in merged if (iv[1] - iv[0]) >= MIN_CLIP_SEC]


# ==========================================
# RENDER (single-pass, no temp clips, no desync)
# ==========================================
def render(intervals, video_file, intro_file, output_file, is_shorts=False):
    """
    Render using FFmpeg trim/atrim + concat filter in ONE pass.
    No intermediate clip files = no chipmunk audio, no A/V desync.
    Supports vertical 9:16 output for Shorts/TikTok.
    """
    work_dir = os.path.dirname(os.path.abspath(output_file)) or "."

    # --- CLEANUP old temp files from previous runs ---
    for old in glob.glob(os.path.join(work_dir, "_part_*.mp4")):
        os.remove(old)
    for old in ["_concat_list.txt", "_merge_list.txt", "_filter.txt"]:
        p = os.path.join(work_dir, old)
        if os.path.exists(p): os.remove(p)

    # Detect main video resolution (intro must match this)
    res_probe = subprocess.run(
        f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height '
        f'-of csv=p=0:s=x "{video_file}"',
        shell=True, capture_output=True, text=True)
    main_res = res_probe.stdout.strip()  # e.g. "1920x1080"
    if not main_res or 'x' not in main_res:
        main_res = "1920x1080"
    main_w, main_h = main_res.split('x')[:2]
    main_w, main_h = int(main_w), int(main_h)

    # Calculate output resolution (Shorts = center-crop to 9:16)
    if is_shorts:
        target_w = int(main_h * 9 / 16)
        # Make even (FFmpeg requires even dimensions)
        target_w = target_w - (target_w % 2)
        target_h = main_h - (main_h % 2)
    else:
        target_w = main_w - (main_w % 2)
        target_h = main_h - (main_h % 2)
    print(f"  Output resolution: {target_w}x{target_h}")

    # Handle intro — always re-encode to match main video resolution
    has_intro = False
    intro_mp4 = None
    if intro_file and os.path.exists(intro_file):
        has_intro = True
        intro_mp4 = os.path.join(work_dir, "_intro_conv.mp4")
        # Scale+crop intro to match exact output resolution
        intro_vf = f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        print(f"  Converting intro to {target_w}x{target_h}...")
        subprocess.run(
            f'ffmpeg -y -i "{intro_file}" '
            f'-vf "{intro_vf}" '
            f'-c:v libx264 -preset fast -c:a aac -ar 48000 '
            f'-loglevel error "{intro_mp4}"', shell=True)

    # --- Build FFmpeg filter graph ---
    # Each interval becomes a trim+atrim pair, then all get concatenated
    filter_lines = []
    concat_inputs = []
    stream_idx = 1 if has_intro else 0  # main video input index
    segment_counter = 0

    # Intro streams (if present) — force exact resolution match
    if has_intro:
        filter_lines.append(f"[0:v]scale={target_w}:{target_h},setsar=1,setpts=PTS-STARTPTS[intro_v];")
        filter_lines.append("[0:a]asetpts=PTS-STARTPTS[intro_a];")
        concat_inputs.append("[intro_v][intro_a]")
        segment_counter = 1

    # Main video segments (with dynamic zoom on high-energy clips)
    for i, (s, e) in enumerate(intervals):
        vid_label = f"v{i}"
        aud_label = f"a{i}"

        # Check if this interval is a high-energy segment (top 20%)
        is_hype = False
        if hasattr(render, '_hype_intervals'):
            for hs, he in render._hype_intervals:
                if s >= hs - 1 and e <= he + 1:
                    is_hype = True
                    break

        clip_dur = e - s

        # Build video filter chain for this segment
        v_filters = f"trim=start={s:.3f}:end={e:.3f},"
        if is_hype:
            v_filters += "scale=iw*1.05:ih*1.05,crop=iw/1.05:ih/1.05,eq=saturation=1.15,"
        if is_shorts:
            v_filters += f"crop={target_w}:{target_h},"
        # Force exact resolution + SAR to match concat requirements
        v_filters += f"scale={target_w}:{target_h},setsar=1,"
        v_filters += f"setpts=PTS-STARTPTS"

        filter_lines.append(
            f"[{stream_idx}:v]{v_filters}[{vid_label}];"
        )
        # Anti-pop micro-fades: 0.05s fade in/out on EACH clip kills click artifacts
        filter_lines.append(
            f"[{stream_idx}:a]atrim=start={s:.3f}:end={e:.3f},"
            f"asetpts=PTS-STARTPTS,"
            f"afade=t=in:st=0:d=0.05,"
            f"afade=t=out:st={max(0, clip_dur - 0.05):.3f}:d=0.05"
            f"[{aud_label}];"
        )
        concat_inputs.append(f"[{vid_label}][{aud_label}]")

    total_segments = len(concat_inputs)

    # Concat all segments
    concat_str = "".join(concat_inputs)
    # Use crossfade-aware concat: acrossfade between audio segments
    # Note: video crossfade in filter_complex requires xfade per pair (complex)
    # so we just do audio crossfade via the acrossfade on the final mixed output
    filter_lines.append(
        f"{concat_str}concat=n={total_segments}:v=1:a=1[outv][outa_raw];"
    )
    # Smooth audio: slight fade in/out on the concatenated result
    filter_lines.append(
        f"[outa_raw]afade=t=in:d=0.1,afade=t=out:st={max(0.1, sum(e-s for s,e in intervals)-0.15):.3f}:d=0.15[outa]"
    )

    filter_graph = "\n".join(filter_lines)

    # Write filter to file (avoids command-line length limits on Windows)
    filter_path = os.path.join(work_dir, "_filter.txt")
    with open(filter_path, "w", encoding="utf-8") as f:
        f.write(filter_graph)

    print(f"  Rendering {len(intervals)} segments in single pass...")

    # Build the FFmpeg command
    inputs = ""
    if has_intro:
        inputs += f'-i "{intro_mp4}" '
    inputs += f'-i "{video_file}"'

    # Auto-detect GPU encoding for speed
    if has_nvenc():
        v_codec = '-c:v h264_nvenc -preset p4 -cq 24 -b:v 8M'
        print("  Using NVIDIA GPU encoder (NVENC) — fast mode")
    else:
        v_codec = '-c:v libx264 -preset ultrafast -crf 23'
        print("  Using CPU encoder (libx264)")

    # Use file-based filter to avoid command-line length limits
    # New FFmpeg uses -/filter_complex, older uses -filter_complex_script
    filter_flag = f'-/filter_complex "{filter_path}"'

    cmd = (
        f'ffmpeg -y {inputs} '
        f'{filter_flag} '
        f'-map "[outv]" -map "[outa]" '
        f'{v_codec} '
        f'-c:a aac -b:a 192k -ar 48000 '
        f'-movflags +faststart '
        f'-loglevel warning -stats '
        f'"{output_file}"'
    )
    # Try rendering — show output to user for progress
    result = subprocess.run(cmd, shell=True)

    # If -/filter_complex not recognized, retry with -filter_complex_script
    if result.returncode != 0:
        print("  Retrying with alternate filter syntax...")
        cmd = cmd.replace(f'-/filter_complex "{filter_path}"',
                          f'-filter_complex_script "{filter_path}"')
        subprocess.run(cmd, shell=True)

    # Cleanup
    if os.path.exists(filter_path):
        os.remove(filter_path)
    if intro_mp4 and "_intro_conv" in intro_mp4 and os.path.exists(intro_mp4):
        os.remove(intro_mp4)

# ==========================================
# ASS KINETIC SUBTITLES (replaces SRT)
# ==========================================
def _ass_time(sec):
    """Convert seconds to ASS timestamp: H:MM:SS.cc"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int((sec % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _colorize_lore(text):
    """Highlight squad names in bold yellow in ASS subtitle format."""
    for name in SQUAD_NAMES:
        # ASS override: bold + yellow highlight, then reset
        text = re.sub(
            rf'\b({re.escape(name)})\b',
            r'{\\b1\\c&H00FFFF&}\1{\\b0\\c&HFFFFFF&}',
            text, flags=re.IGNORECASE
        )
    return text


def generate_subtitles(segments, kept_intervals, output_path, is_shorts=False):
    """Generate styled ASS subtitles + plain SRT. Squad names highlighted yellow."""
    # Build subtitle entries remapped to output timeline
    entries = []
    output_offset = 0.0
    for iv_start, iv_end in kept_intervals:
        for seg in segments:
            text = seg["text"].strip()
            if not text or seg["end"] <= iv_start or seg["start"] >= iv_end:
                continue
            clamp_s = max(seg["start"], iv_start)
            clamp_e = min(seg["end"], iv_end)
            out_s = output_offset + (clamp_s - iv_start)
            out_e = output_offset + (clamp_e - iv_start)
            if out_e - out_s >= 0.1:
                entries.append((out_s, out_e, text))
        output_offset += (iv_end - iv_start)

    # --- ASS file (styled subtitles) ---
    ass_path = output_path.replace(".mp4", ".ass")
    res_x = 1080 if is_shorts else 1920
    res_y = 1920 if is_shorts else 1080
    font_size = 48 if is_shorts else 36
    margin_v = 120 if is_shorts else 60

    ass_header = f"""[Script Info]
Title: Vlog Automaton Subtitles
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,20,20,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header)
        for s, e, text in entries:
            styled_text = _colorize_lore(text)
            f.write(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Default,,0,0,0,,{styled_text}\n")

    # --- SRT file (plain, for YouTube upload) ---
    srt_path = output_path.replace(".mp4", ".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (s, e, text) in enumerate(entries, 1):
            sh, sm, ss, sms = int(s//3600), int((s%3600)//60), int(s%60), int((s%1)*1000)
            eh, em, es, ems = int(e//3600), int((e%3600)//60), int(e%60), int((e%1)*1000)
            f.write(f"{i}\n{sh:02d}:{sm:02d}:{ss:02d},{sms:03d} --> {eh:02d}:{em:02d}:{es:02d},{ems:03d}\n{text}\n\n")

    print(f"  {len(entries)} subtitle entries generated")
    print(f"  ASS (styled): {ass_path}")
    print(f"  SRT (YouTube): {srt_path}")


# ==========================================
# UI + MAIN
# ==========================================
def banner():
    print()
    print("=" * 56)
    print("     VLOG AUTOMATON v9.0 — Content Creator Edition")
    print("     Shorts Mode | Kinetic Subs | Genre-Tuned")
    print("=" * 56)

def pick_genre():
    print()
    print("  What type of content?")
    print("  1) DISCORD CALL  — Banter & lore focused")
    print("  2) GAMING        — Roblox/Valorant/Minecraft")
    print("  3) VLOG          — IRL, talking head")
    print("  4) MIXED/AUTO    — Let the AI decide")
    while True:
        try:
            g = int(input("\n  Pick genre (1/2/3/4): "))
            if g in GENRE:
                print(f"  -> {GENRE[g]['name']}")
                return g
        except: pass
        print("  Enter 1, 2, 3, or 4")

def pick_quality():
    print()
    print("  1) HIGHLIGHTS  — Fast-paced, only bangers     (~25-45%)")
    print("  2) BALANCED    — Entertaining, keeps flow     (~45-70%)")
    print("  3) CHILL       — Trims dead air & boring      (~70-90%)")
    while True:
        try:
            q = int(input("\n  Pick quality (1/2/3): "))
            if q in QUALITY:
                print(f"  -> {QUALITY[q]['name']}")
                return q
        except: pass
        print("  Enter 1, 2, or 3")

def get_inputs():
    if len(sys.argv) > 1:
        video  = sys.argv[1].strip('"').strip()
        intro  = sys.argv[2].strip('"').strip() if len(sys.argv) > 2 else ""
        output = sys.argv[3].strip()            if len(sys.argv) > 3 else ""
        qual   = int(sys.argv[4])               if len(sys.argv) > 4 else None
    else:
        print("\n  Tip: drag a FOLDER for multi-file vlog, or a single video file")
        print("  Tip: for multiple files, separate with commas\n")
        video  = input("  Video path(s): ").strip('"').strip()
        intro  = input("  Intro path (Enter to skip): ").strip('"').strip()
        output = input("  Output name (Enter = vlog_cut.mp4): ").strip()
        qual   = None
    if not output: output = "vlog_cut.mp4"
    if not output.endswith(".mp4"): output += ".mp4"
    return video, intro, output, qual


def main():
    banner()
    raw_video, intro_file, output_file, quality = get_inputs()

    # --- Resolve & merge multiple files ---
    video_files = resolve_inputs(raw_video)
    if not video_files:
        print(f"\n  No video files found: {raw_video}")
        return

    work_dir = os.path.dirname(os.path.abspath(output_file)) or "."

    # Show input duration
    total_input_dur = sum(get_duration(vf) for vf in video_files)
    print(f"\n  Input: {len(video_files)} file(s), {fmt(total_input_dur)} total")

    if quality is None:
        quality = pick_quality()

    genre = pick_genre()
    genre_weights = GENRE[genre]["weights"]
    print(f"  Scoring weights: transcript={genre_weights[0]}, audio={genre_weights[1]}, visual={genre_weights[2]}")

    shorts_input = input("\n  Export as vertical 9:16 Shorts/TikTok? (y/n): ").strip().lower()
    is_shorts = shorts_input == 'y'
    if is_shorts:
        print("  -> Vertical 9:16 mode enabled")

    print("\n  All set! Running pipeline...\n")

    # === Phase 1: Whisper Transcription ===
    # Transcribe EACH file separately to avoid OOM on large merges
    print(f"\n{'='*56}")
    print(f"  Phase 1/6 — Whisper Transcription")
    print(f"{'='*56}")

    whisper_size = "small"
    try:
        model = WhisperModel(whisper_size, device="cuda", compute_type="float16")
        print(f"  GPU active | model: {whisper_size}")
    except:
        model = WhisperModel(whisper_size, device="cpu", compute_type="int8")
        print(f"  CPU mode | model: {whisper_size}")

    segments = []
    time_offset = 0.0
    for fi, vf in enumerate(video_files):
        dur = get_duration(vf)
        fname = os.path.basename(vf)
        print(f"  Transcribing [{fi+1}/{len(video_files)}] {fname} ({fmt(dur)})...")
        segs_raw, info = model.transcribe(vf, vad_filter=True, language="en")
        file_segs = [{"start": s.start + time_offset, "end": s.end + time_offset, "text": s.text}
                     for s in segs_raw]
        segments.extend(file_segs)
        print(f"    {len(file_segs)} segments")
        time_offset += dur
        sys.stdout.flush()

    print(f"  [DEBUG] Transcription loop complete. {len(segments)} total segments.")
    sys.stdout.flush()

    # Don't explicitly free the model — CUDA cleanup can hard-crash the process
    # Python will handle it when the process exits

    if not segments:
        print("  No speech detected. Aborting.")
        return

    speech_t = sum(s["end"] - s["start"] for s in segments)
    print(f"  Total: {len(segments)} segments, {fmt(speech_t)} of speech")
    sys.stdout.flush()

    # Show transcript sample so user can verify it's not garbage
    print(f"  Sample: \"{segments[0]['text'].strip()[:60]}...\"")
    if len(segments) > 10:
        mid = len(segments) // 2
        print(f"  Sample: \"{segments[mid]['text'].strip()[:60]}...\"")

    # Now merge video files (for rendering later)
    video_file, is_merged = merge_video_files(video_files, work_dir)
    total_dur = get_duration(video_file)

    # === Phase 2: Audio Loudness ===
    print(f"\n{'='*56}")
    print(f"  Phase 2/6 — Audio Loudness Analysis")
    print(f"{'='*56}")
    loudness = analyze_loudness(video_file)

    rms_min, rms_range = 0, 1
    if loudness:
        vals = [r for _, r in loudness if r > -100]
        if vals:
            rms_min = min(vals)
            rms_range = max(vals) - rms_min
            if rms_range == 0: rms_range = 1

    # === Phase 3: Visual Motion ===
    print(f"\n{'='*56}")
    print(f"  Phase 3/6 — Visual Motion Analysis")
    print(f"{'='*56}")
    motion_data = analyze_motion(video_file)

    # === Phase 4: Topic-Aware Scoring ===
    print(f"\n{'='*56}")
    print(f"  Phase 4/6 — Topic-Aware Content Scoring")
    print(f"{'='*56}")

    topics = build_topics(segments)
    print(f"  Detected {len(topics)} conversation topics")

    topic_scores = []
    for topic in tqdm(topics, desc="  Scoring topics"):
        sc = score_topic(topic, loudness, rms_min, rms_range, motion_data, genre_weights)
        topic_scores.append(sc)

    keep_intervals = select_topics(topics, topic_scores, total_dur, quality)

    if not keep_intervals:
        print("  Scoring produced no clips. Aborting.")
        return

    keep_dur = sum(e - s for s, e in keep_intervals)
    cut_dur = total_dur - keep_dur
    pct = (keep_dur / total_dur * 100) if total_dur else 0

    # --- Topic Preview (see what gets kept/cut) ---
    preset = QUALITY[quality]
    scores_sorted = sorted(topic_scores)
    idx = int(len(scores_sorted) * preset["pct"] / 100)
    threshold = scores_sorted[min(idx, len(scores_sorted) - 1)]

    print(f"\n  {'#':<4} {'Time':<16} {'Duration':<9} {'Score':<8} {'Status':<8} Preview")
    print(f"  {'—'*75}")
    for i, (topic, sc) in enumerate(zip(topics, topic_scores)):
        t_start = topic[0]["start"]
        t_end = topic[-1]["end"]
        dur = t_end - t_start
        status = "KEEP" if sc >= threshold else "CUT"
        # Show first few words of the topic
        preview = " ".join(s["text"].strip() for s in topic[:3])[:45]
        color = "" if status == "KEEP" else ""
        print(f"  {i+1:<4} {fmt(t_start)}-{fmt(t_end):<11} {dur:>5.1f}s   {sc:>6.1f}  {status:<8} {preview}...")

    print(f"\n  Input:   {fmt(total_dur)}")
    print(f"  Output:  {fmt(keep_dur)}  ({pct:.0f}% kept)")
    print(f"  Cut:     {fmt(cut_dur)}  ({100-pct:.0f}% removed)")
    print(f"  Clips:   {len(keep_intervals)}")
    print(f"  Style:   {QUALITY[quality]['name']}")

    # --- Save edit decision log (with full transcript for review UI) ---
    import json
    log_path = output_file.replace(".mp4", "_edit_log.json")
    edit_log = {
        "input": os.path.basename(video_file),
        "input_path": os.path.abspath(video_file),
        "output": output_file,
        "quality": QUALITY[quality]["name"],
        "input_duration": round(total_dur, 2),
        "output_duration": round(keep_dur, 2),
        "pct_kept": round(pct, 1),
        "topics": [
            {
                "id": i+1,
                "start": round(topic[0]["start"], 2),
                "end": round(topic[-1]["end"], 2),
                "score": round(sc, 1),
                "status": "KEEP" if sc >= threshold else "CUT",
                "transcript": " ".join(s["text"].strip() for s in topic),
                "segments": [
                    {"start": round(s["start"], 2),
                     "end": round(s["end"], 2),
                     "text": s["text"].strip()}
                    for s in topic
                ],
            }
            for i, (topic, sc) in enumerate(zip(topics, topic_scores))
        ],
        "intervals_kept": [[round(s,2), round(e,2)] for s, e in keep_intervals],
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(edit_log, f, indent=2, ensure_ascii=False)
    print(f"\n  Edit log saved: {log_path}")

    # === Phase 5: Render ===
    print(f"\n{'='*56}")
    print(f"  Phase 5/6 — FFmpeg Rendering")
    print(f"{'='*56}")

    # Tag high-energy intervals for dynamic zoom
    # Top 20% scoring topics get the zoom treatment
    scores_sorted_all = sorted(topic_scores, reverse=True)
    hype_threshold = scores_sorted_all[max(0, len(scores_sorted_all) // 5 - 1)] if scores_sorted_all else 100
    hype_intervals = []
    for topic, sc in zip(topics, topic_scores):
        if sc >= hype_threshold:
            hype_intervals.append((topic[0]["start"], topic[-1]["end"]))
    render._hype_intervals = hype_intervals
    print(f"  {len(hype_intervals)} high-energy segments tagged for zoom")

    render(keep_intervals, video_file, intro_file, output_file, is_shorts)

    # Cleanup merged temp file
    if is_merged and os.path.exists(video_file):
        os.remove(video_file)

    # === Phase 6: Kinetic Subtitles ===
    print(f"\n{'='*56}")
    print(f"  Phase 6/6 — Kinetic Subtitles")
    print(f"{'='*56}")

    generate_subtitles(segments, keep_intervals, output_file, is_shorts)

    ass_path = output_file.replace(".mp4", ".ass")
    srt_path = output_file.replace(".mp4", ".srt")
    print(f"\n{'='*56}")
    print(f"  DONE: {output_file}")
    print(f"  {fmt(total_dur)} -> {fmt(keep_dur)} ({pct:.0f}% kept)")
    print(f"  Subtitles: {ass_path} (styled) + {srt_path} (YouTube)")
    print(f"  Edit log:  {log_path}")
    print(f"  Review UI: review.html")
    if is_shorts:
        print(f"  Format:    9:16 Vertical (Shorts/TikTok)")
    print(f"{'='*56}")


def render_from_json(json_path):
    """Render from an adjusted edit log JSON (from the review UI)."""
    import json
    print(f"\n{'='*56}")
    print(f"  Rendering from adjusted edit log")
    print(f"{'='*56}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_file = data.get("input_path", data.get("input", ""))
    if not os.path.exists(video_file):
        # Try same directory as JSON
        video_file = os.path.join(os.path.dirname(json_path), data.get("input", ""))
    if not os.path.exists(video_file):
        print(f"  Video not found: {video_file}")
        vid = input("  Enter video path: ").strip('"').strip()
        if os.path.exists(vid):
            video_file = vid
        else:
            print("  Aborting.")
            return

    intervals = [tuple(iv) for iv in data["intervals_kept"]]
    output = data.get("output", "adjusted_output.mp4")

    total_dur = data.get("input_duration", 0)
    keep_dur = sum(e - s for s, e in intervals)
    pct = (keep_dur / total_dur * 100) if total_dur else 0

    print(f"  Video:   {os.path.basename(video_file)}")
    print(f"  Output:  {output}")
    print(f"  Clips:   {len(intervals)}")
    print(f"  Keep:    {fmt(keep_dur)} ({pct:.0f}%)")

    render(intervals, video_file, "", output)

    print(f"\n{'='*56}")
    print(f"  DONE: {output}")
    print(f"{'='*56}")


# ==========================================
# MANUAL TRIM MODE
# ==========================================
def parse_time(t_str):
    """Parse time string like '1:30', '0:05:30', or '90' into seconds."""
    t_str = t_str.strip()
    parts = t_str.split(':')
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        else:
            return float(t_str)
    except ValueError:
        return None


def manual_trim():
    """Let the user manually specify timestamp ranges to keep."""
    print()
    print("=" * 56)
    print("     VLOG AUTOMATON v9.0 — Manual Trim Mode")
    print("     Cut exact timestamps from your video")
    print("=" * 56)

    print("\n  Tip: drag a video file\n")
    video = input("  Video path: ").strip('"').strip()
    if not os.path.exists(video):
        print(f"  File not found: {video}")
        return

    intro = input("  Intro path (Enter to skip): ").strip('"').strip()
    output = input("  Output name (Enter = trimmed.mp4): ").strip() or "trimmed.mp4"
    if not output.endswith(".mp4"): output += ".mp4"

    total_dur = get_duration(video)
    print(f"\n  Video duration: {fmt(total_dur)}")

    print("\n  Enter timestamps to KEEP (ranges you want in the final video)")
    print("  Format: start-end, start-end, ...")
    print("  Time format: M:SS or H:MM:SS or seconds")
    print("  Example: 0:30-2:00, 5:10-7:30, 12:00-15:45\n")

    raw = input("  Timestamps: ").strip()
    if not raw:
        print("  No timestamps entered. Aborting.")
        return

    # Parse timestamp ranges
    intervals = []
    for part in raw.split(","):
        part = part.strip()
        if "-" not in part:
            print(f"  Invalid range: '{part}' (need start-end)")
            continue
        start_str, end_str = part.split("-", 1)
        start = parse_time(start_str)
        end = parse_time(end_str)
        if start is None or end is None:
            print(f"  Could not parse: '{part}'")
            continue
        if end <= start:
            print(f"  Skipping: '{part}' (end <= start)")
            continue
        if start > total_dur:
            print(f"  Skipping: '{part}' (start beyond video)")
            continue
        end = min(end, total_dur)
        intervals.append((start, end))

    if not intervals:
        print("  No valid intervals. Aborting.")
        return

    intervals.sort(key=lambda x: x[0])

    # Shorts mode
    shorts_input = input("\n  Export as vertical 9:16 Shorts/TikTok? (y/n): ").strip().lower()
    is_shorts = shorts_input == 'y'

    keep_dur = sum(e - s for s, e in intervals)
    print(f"\n  Keeping {len(intervals)} segment(s), {fmt(keep_dur)} total:")
    for i, (s, e) in enumerate(intervals, 1):
        print(f"    {i}. {fmt(s)} -> {fmt(e)}  ({e-s:.1f}s)")

    print(f"\n{'='*56}")
    print(f"  Rendering...")
    print(f"{'='*56}")

    render(intervals, video, intro, output, is_shorts)

    # Generate subtitles if we can transcribe
    print(f"\n  Generating subtitles...")
    try:
        whisper_size = "small"
        try:
            model = WhisperModel(whisper_size, device="cuda", compute_type="float16")
        except:
            model = WhisperModel(whisper_size, device="cpu", compute_type="int8")
        segs_raw, info = model.transcribe(video, vad_filter=True, language="en")
        segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segs_raw]
        generate_subtitles(segments, intervals, output, is_shorts)
    except Exception as e:
        print(f"  Subtitle generation skipped: {e}")

    print(f"\n{'='*56}")
    print(f"  DONE: {output}")
    print(f"  {fmt(total_dur)} -> {fmt(keep_dur)} ({keep_dur/total_dur*100:.0f}% kept)")
    print(f"{'='*56}")


if __name__ == "__main__":
    # Check if first arg is a JSON file (render-from-adjusted mode)
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        render_from_json(sys.argv[1])
    else:
        try:
            print()
            print("=" * 56)
            print("     VLOG AUTOMATON v9.0 — Content Creator Edition")
            print("=" * 56)
            print("\n  1) AUTO-CUT    — AI picks the best moments")
            print("  2) MANUAL TRIM — You pick exact timestamps")
            while True:
                try:
                    mode = int(input("\n  Mode (1/2): "))
                    if mode in (1, 2): break
                except: pass
                print("  Enter 1 or 2")

            if mode == 1:
                main()
            else:
                manual_trim()
        except Exception as e:
            import traceback
            print(f"\n  ERROR: {e}")
            traceback.print_exc()
            input("\n  Press Enter to exit...")