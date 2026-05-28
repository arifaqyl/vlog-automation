"""
notify.py — optional Telegram completion notification for vlog-automation.

Usage: import notify; notify.done(stats_dict) at end of processing.
Set TG_BOT_TOKEN and TG_CHAT_ID env vars, or hardcode below.
Leave blank to disable.
"""

import os
import json
import requests
from datetime import datetime

# ── CONFIG ───────────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")   # e.g. 123456:ABC...
CHAT_ID   = os.getenv("TG_CHAT_ID",   "")   # your Telegram user/chat ID
STATS_LOG = "processing_log.json"

# ─────────────────────────────────────────────────────────────────────────────

def _send(text):
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=8
        )
    except Exception:
        pass


def _save_stats(stats):
    """Append this run to processing_log.json."""
    try:
        history = []
        if os.path.exists(STATS_LOG):
            with open(STATS_LOG, "r", encoding="utf-8") as f:
                history = json.load(f)
        history.append(stats)
        with open(STATS_LOG, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


def _fmt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def done(stats: dict):
    """
    Call at end of processing. stats keys:
      input_file    str
      total_dur     float  (seconds)
      kept_dur      float  (seconds)
      output_file   str
      genre         str  (optional)
      quality       str  (optional)
    """
    stats["timestamp"] = datetime.now().isoformat()
    _save_stats(stats)

    total = stats.get("total_dur", 0)
    kept  = stats.get("kept_dur",  0)
    pct   = round(kept / total * 100) if total else 0
    cut   = total - kept
    fname = os.path.basename(stats.get("output_file", "output.mp4"))
    genre = stats.get("genre", "")
    qual  = stats.get("quality", "")

    msg = (
        f"<b>vlog-automation done</b>\n"
        f"Output: {fname}\n"
        f"{_fmt(total)} → {_fmt(kept)}  ({pct}% kept, {_fmt(cut)} cut)\n"
    )
    if genre or qual:
        msg += f"Mode: {genre} / {qual}\n"

    _send(msg)

    # Also print to terminal
    print(f"\n  Notification sent ({fname})")
