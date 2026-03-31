"""
segment_utils.py  —  Shared utilities for C1 / C2 / C3 chunkers
=================================================================
Import this in every chunker script. Do not run directly.

Provides:
  - ocr_confidence(ocr_text)              → float 0–1
  - is_ocr_failed(ocr_text)               → bool
  - classify_chunk(transcript, ocr_text)  → (content_type, is_code)
  - ocr_jaccard(ocr_a, ocr_b)             → float 0–1  [C3 only]
  - build_segment_id(...)                 → str
  - build_deep_link(video_id, start_sec)  → str
  - load_multimodal(path)                 → list[dict]
  - load_metadata(path)                   → dict | None
  - iter_lectures(output_dir, courses)    → yields (metadata, segments)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

# ── thresholds (edit here if you want to tune) ──────────────────────────────
OCR_CONFIDENCE_THRESHOLD = 0.15   # below this → ocr_failed = True
OCR_JACCARD_THRESHOLD    = 0.30   # below this → slide changed  (C3)
MIN_REAL_WORD_LEN        = 3      # min chars to count as a real word in OCR
CODE_SIGNAL_MIN          = 2      # need this many code signals to flag as code

# ── code signal patterns (checked against transcript + ocr combined) ─────────
_CODE_SIGNALS = [
    r'\bdef ',        r'\bclass ',      r'\bimport ',
    r'\bfor .{1,30} in ', r'\bif .{1,30}:',  r'\breturn ',
    r'\bprint\(',     r'\bwhile ',      r'\bint\b',
    r'\bvoid\b',      r'#include',      r'\$gcc',
    r'\$\.\/',        r'\bchar\b',      r'\bprintf\(',
    r'\bstruct\b',    r'\bpublic\b',    r'\bprivate\b',
    r'[a-z_]+\([^)]{0,40}\)\s*[{:]',   # function call/def pattern
    r'=\s*\[',        r'=\s*\{',        # list/dict literals
]

# ── theoretical keyword set ──────────────────────────────────────────────────
_THEORY_KEYWORDS = {
    'theorem', 'proof', 'definition', 'formally', 'algorithm',
    'complexity', 'lemma', 'corollary', 'proposition', 'analysis',
    'recurrence', 'induction', 'invariant', 'asymptotic',
}


# ─────────────────────────────────────────────────────────────────────────────
# OCR quality assessment
# ─────────────────────────────────────────────────────────────────────────────

def ocr_confidence(ocr_text: str) -> float:
    """
    Returns a float 0–1 representing OCR quality.

    Computed as:
        real_words / max(total_chars / 5, 1)

    Where a "real word" is a token of 3+ alphabetic characters.
    A clean text slide should score > 0.4.
    A diagram-only slide will score < 0.15.
    An empty string scores 0.0.
    """
    if not ocr_text or not ocr_text.strip():
        return 0.0
    total_chars = len(ocr_text.replace("\n", "").replace(" ", ""))
    if total_chars == 0:
        return 0.0
    real_words = re.findall(r'[a-zA-Z]{3,}', ocr_text)
    # normalise against expected word count assuming ~5 chars/word
    score = len(real_words) / max(total_chars / 5, 1)
    return round(min(score, 1.0), 4)


def is_ocr_failed(ocr_text: str) -> bool:
    """Returns True when OCR confidence is below the usable threshold."""
    return ocr_confidence(ocr_text) < OCR_CONFIDENCE_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Content classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_chunk(transcript: str, ocr_text: str) -> tuple[str, bool]:
    """
    Returns (content_type, is_code_segment).

    content_type is one of:
        "code"        — slide/transcript contains programming syntax
        "theoretical" — contains maths/proof/algorithm keywords
        "conceptual"  — everything else

    is_code_segment mirrors content_type == "code".
    """
    combined = (ocr_text or "") + " " + (transcript or "")

    # ── code check ────────────────────────────────────────────────────────
    code_hits = sum(
        1 for pattern in _CODE_SIGNALS
        if re.search(pattern, combined)
    )
    if code_hits >= CODE_SIGNAL_MIN:
        return "code", True

    # ── theoretical check ─────────────────────────────────────────────────
    lower = combined.lower()
    theory_hits = sum(1 for kw in _THEORY_KEYWORDS if kw in lower)
    if theory_hits >= 1:
        return "theoretical", False

    return "conceptual", False


# ─────────────────────────────────────────────────────────────────────────────
# Slide similarity  (C3 chunker only)
# ─────────────────────────────────────────────────────────────────────────────

def ocr_jaccard(ocr_a: str, ocr_b: str) -> float:
    """
    Jaccard similarity between two OCR texts based on their word sets.

    Uses only alphabetic tokens of length >= MIN_REAL_WORD_LEN to filter
    out single-character OCR noise.

    Returns:
        1.0  if both are empty  (same slide — no text visible)
        0.0  if exactly one is empty  (slide appeared or disappeared)
        Jaccard score otherwise

    A score below OCR_JACCARD_THRESHOLD means the slide changed.
    """
    def _word_set(text: str) -> set[str]:
        return set(re.findall(rf'[a-zA-Z]{{{MIN_REAL_WORD_LEN},}}', text.lower()))

    words_a = _word_set(ocr_a or "")
    words_b = _word_set(ocr_b or "")

    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union        = words_a | words_b
    return round(len(intersection) / len(union), 4)


# ─────────────────────────────────────────────────────────────────────────────
# ID and URL builders
# ─────────────────────────────────────────────────────────────────────────────

def build_segment_id(
    course_id:    str,
    lecture_num:  int,
    strategy:     str,   # "c1", "c2", "c3"
    chunk_num:    int,
) -> str:
    """
    Returns a globally unique, human-readable segment ID.
    Example: "dbms-lec004-c1-007"
    """
    return f"{course_id}-lec{lecture_num:03d}-{strategy}-{chunk_num:03d}"


def build_deep_link(video_id: str, start_sec: float) -> str:
    """
    Returns a YouTube URL that starts playback at start_sec.
    Example: https://www.youtube.com/watch?v=SFcKedpnqwg&t=743s
    """
    t = max(0, int(start_sec))
    return f"https://www.youtube.com/watch?v={video_id}&t={t}s"


# ─────────────────────────────────────────────────────────────────────────────
# File loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_multimodal(path: Path) -> list[dict]:
    """
    Loads multimodal.json and normalises every record to have:
        text, start, end, duration, ocr_text

    The Pipeline B script stores 'end' not 'duration'.
    This function adds 'duration = end - start' if missing.
    Returns empty list on any error.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        normalised = []
        for seg in raw:
            start    = float(seg.get("start", 0))
            end      = float(seg.get("end",   start))
            duration = float(seg.get("duration", end - start))
            normalised.append({
                "text":     seg.get("text", "").strip(),
                "start":    round(start,    3),
                "end":      round(end,      3),
                "duration": round(duration, 3),
                "ocr_text": seg.get("ocr_text", "").strip(),
            })
        return normalised
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return []


def load_metadata(lecture_dir: Path) -> dict | None:
    """
    Loads metadata.json from a lecture directory.
    Returns None if the file does not exist or is malformed.
    """
    meta_path = lecture_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [WARN] Could not load {meta_path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Lecture iterator — used by all chunkers
# ─────────────────────────────────────────────────────────────────────────────

def iter_lectures(
    output_dir: Path,
    course_ids: list[str] | None = None,
) -> Iterator[tuple[dict, list[dict]]]:
    """
    Walks output_dir and yields (metadata, segments) for every lecture
    that has both metadata.json and multimodal.json present.

    Args:
        output_dir  — root of Pipeline B output (e.g. Path("output"))
        course_ids  — if given, only yield lectures from these course IDs

    Yields:
        metadata  — dict from metadata.json
        segments  — list of normalised segment dicts from multimodal.json

    Skips and warns on:
        - Course folders not found in course_ids filter
        - Lecture folders missing metadata.json  (metadata_builder not run)
        - Lecture folders missing multimodal.json
        - multimodal.json that is empty after loading
    """
    if not output_dir.exists():
        print(f"[ERROR] output_dir not found: {output_dir}")
        return

    for course_dir in sorted(output_dir.iterdir()):
        if not course_dir.is_dir():
            continue
        course_id = course_dir.name

        if course_ids and course_id not in course_ids:
            continue

        lecture_dirs = sorted(
            d for d in course_dir.iterdir() if d.is_dir()
        )

        for lecture_dir in lecture_dirs:
            # ── load metadata ──────────────────────────────────────────
            metadata = load_metadata(lecture_dir)
            if metadata is None:
                print(f"  [SKIP] No metadata.json in {lecture_dir} "
                      f"— run metadata_builder.py first")
                continue

            # ── load segments ──────────────────────────────────────────
            mm_path  = lecture_dir / "multimodal.json"
            if not mm_path.exists():
                print(f"  [SKIP] No multimodal.json in {lecture_dir}")
                continue

            segments = load_multimodal(mm_path)
            if not segments:
                print(f"  [SKIP] Empty multimodal.json in {lecture_dir}")
                continue

            yield metadata, segments
