"""
chunker_c3.py  —  Strategy C3: Slide-Boundary Chunking  (Novel Contribution)
==============================================================================
Detects slide changes by computing Jaccard word-overlap similarity between
consecutive segments' OCR text. When similarity drops below a threshold,
the slide has changed and a new chunk begins.

Why this is the right approach for lecture video retrieval:
  A lecture slide represents the professor's deliberate decision to present
  one concept. Advancing the slide is an explicit semantic boundary signal.
  C3 respects this structure; C1 and C2 ignore it entirely.

The OCR Jaccard method:
  For each consecutive segment pair (N, N+1):
    similarity = |words(ocr_N) ∩ words(ocr_N+1)| / |words(ocr_N) ∪ words(ocr_N+1)|
  If similarity < SLIDE_CHANGE_THRESHOLD → new chunk boundary

Fallback for low-quality OCR:
  When a segment's OCR is empty or confidence < threshold, OCR-based
  boundary detection is unreliable. For those segments the script falls
  back to C2 (word-count) boundaries and flags the chunk as
  "boundary_method": "ocr_fallback_c2".

Min/max duration rules:
  min_sec: If a detected chunk is shorter than this, merge with the next.
           Handles brief slide flashes during transitions.
  max_sec: If a chunk exceeds this, split at the nearest sentence boundary
           within the window. Handles slides displayed for very long periods.

Usage
-----
    python chunker_c3.py                        # all courses, defaults
    python chunker_c3.py --course os            # one course
    python chunker_c3.py --threshold 0.25 --suffix _t025      # more sensitive to slide changes
    python chunker_c3.py --threshold 0.40 --suffix _t040    # less sensitive
    python chunker_c3.py --dry-run              # stats only

Output
------
    data/processed/segments_c3.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ── load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _here = Path(__file__).resolve().parent
    for _c in [_here, _here.parent, _here.parent.parent]:
        if (_c / ".env").exists():
            load_dotenv(_c / ".env")
            break
except ImportError:
    pass

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))
OUTPUT_DIR   = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "output"))
PROCESSED    = PROJECT_ROOT / "data" / "processed"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_utils import (
    build_segment_id,
    build_deep_link,
    classify_chunk,
    ocr_confidence,
    is_ocr_failed,
    ocr_jaccard,
    iter_lectures,
    OCR_CONFIDENCE_THRESHOLD,
    OCR_JACCARD_THRESHOLD,
)

# ── config ────────────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = OCR_JACCARD_THRESHOLD   # 0.30
MIN_CHUNK_SEC     = 15     # merge chunks shorter than this
MAX_CHUNK_SEC     = 120    # split chunks longer than this
FALLBACK_WORDS    = 200    # word-count fallback when OCR is unreliable
STRATEGY_ID       = "c3"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_ocr_reliable(seg: dict) -> bool:
    """Returns True when a segment's OCR text is good enough for Jaccard."""
    return ocr_confidence(seg.get("ocr_text", "")) >= OCR_CONFIDENCE_THRESHOLD


def _split_at_sentence(window_segs: list[dict], max_sec: float) -> list[list[dict]]:
    """
    Splits window_segs into sub-windows not exceeding max_sec.
    Boundaries are placed at segment boundaries (never mid-sentence).
    Returns a list of segment-lists.
    """
    if not window_segs:
        return []

    result       = []
    current      = []
    window_start = window_segs[0]["start"]

    for seg in window_segs:
        current.append(seg)
        if seg["end"] - window_start >= max_sec:
            result.append(current)
            current      = []
            window_start = seg["end"]

    if current:
        result.append(current)

    return result


def _flush(
    window_segs:     list[dict],
    metadata:        dict,
    chunk_num:       int,
    boundary_method: str = "ocr_jaccard",
) -> dict:
    """Converts accumulated segments into one chunk record."""
    course_id   = metadata.get("course_id", "unknown")
    video_id    = metadata.get("video_id", "")
    lec_num     = metadata.get("lecture_number", 0)
    lec_title   = metadata.get("lecture_title", metadata.get("lecture_folder", ""))
    course_name = metadata.get("course_name", "")
    instructor  = metadata.get("instructor", "")
    institute   = metadata.get("institute", "")

    transcript = " ".join(s["text"] for s in window_segs).strip()

    # deduplicate consecutive identical OCR blocks
    ocr_parts = []
    prev_ocr  = None
    for s in window_segs:
        ocr = s.get("ocr_text", "").strip()
        if ocr and ocr != prev_ocr:
            ocr_parts.append(ocr)
            prev_ocr = ocr
    ocr_text = "\n---\n".join(ocr_parts)

    start_sec = window_segs[0]["start"]
    end_sec   = window_segs[-1]["end"]
    dur_sec   = round(end_sec - start_sec, 2)
    wc        = len(transcript.split())
    conf      = ocr_confidence(ocr_text)
    failed    = is_ocr_failed(ocr_text)
    ctype, is_code = classify_chunk(transcript, ocr_text)

    return {
        "segment_id":        build_segment_id(course_id, lec_num,
                                              STRATEGY_ID, chunk_num),
        "course_id":         course_id,
        "course_name":       course_name,
        "instructor":        instructor,
        "institute":         institute,
        "lecture_title":     lec_title,
        "lecture_number":    lec_num,
        "youtube_url":       metadata.get("youtube_url", ""),
        "youtube_deep_link": build_deep_link(video_id, start_sec),
        "start_sec":         round(start_sec, 2),
        "end_sec":           round(end_sec, 2),
        "duration_sec":      dur_sec,
        "transcript":        transcript,
        "ocr_text":          ocr_text,
        "ocr_confidence":    conf,
        "ocr_failed":        failed,
        "is_code_segment":   is_code,
        "content_type":      ctype,
        "chunking_strategy": "C3_slide_boundary",
        "boundary_method":   boundary_method,
        "word_count":        wc,
        "source_pipeline":   metadata.get("source_pipeline", "whisper_ocr"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# C3 chunking logic
# ─────────────────────────────────────────────────────────────────────────────

def chunk_lecture_c3(
    segments:   list[dict],
    metadata:   dict,
    threshold:  float = DEFAULT_THRESHOLD,
) -> list[dict]:
    """
    Splits a lecture's segments using slide-boundary detection.

    For each consecutive segment pair the OCR Jaccard similarity is computed.
    A low similarity score means the slide changed → chunk boundary.

    When OCR is unreliable for a segment, falls back to word-count (C2-style)
    boundary detection for that portion.

    Returns:
        List of chunk dicts ready to write to segments_c3.jsonl
    """
    if not segments:
        return []

    raw_windows: list[tuple[list[dict], str]] = []  # (segs, boundary_method)
    window_segs  = [segments[0]]
    word_count   = len(segments[0]["text"].split())
    boundary_met = "ocr_jaccard"

    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]

        # ── decide whether to use OCR or word-count boundary ─────────
        prev_reliable = _is_ocr_reliable(prev)
        curr_reliable = _is_ocr_reliable(curr)

        if prev_reliable and curr_reliable:
            # Both have good OCR — use Jaccard similarity
            sim = ocr_jaccard(
                prev.get("ocr_text", ""),
                curr.get("ocr_text", ""),
            )
            slide_changed = sim < threshold
            method = "ocr_jaccard"
        else:
            # OCR unreliable — fall back to word-count
            slide_changed = word_count >= FALLBACK_WORDS
            method = "ocr_fallback_c2"

        if slide_changed:
            raw_windows.append((window_segs, boundary_met))
            window_segs  = [curr]
            word_count   = len(curr["text"].split())
            boundary_met = method
        else:
            window_segs.append(curr)
            word_count += len(curr["text"].split())

    if window_segs:
        raw_windows.append((window_segs, boundary_met))

    # ── apply min/max duration rules ──────────────────────────────────
    # Pass 1: merge windows shorter than MIN_CHUNK_SEC with the next window
    merged_windows: list[tuple[list[dict], str]] = []
    i = 0
    while i < len(raw_windows):
        segs, meth = raw_windows[i]
        dur = segs[-1]["end"] - segs[0]["start"]

        if dur < MIN_CHUNK_SEC and i + 1 < len(raw_windows):
            # merge with next window
            next_segs, next_meth = raw_windows[i + 1]
            raw_windows[i + 1] = (segs + next_segs, meth)
            i += 1
            continue

        merged_windows.append((segs, meth))
        i += 1

    # Pass 2: split windows longer than MAX_CHUNK_SEC at sentence boundaries
    final_windows: list[tuple[list[dict], str]] = []
    for segs, meth in merged_windows:
        dur = segs[-1]["end"] - segs[0]["start"]
        if dur > MAX_CHUNK_SEC:
            sub_windows = _split_at_sentence(segs, MAX_CHUNK_SEC)
            for sub in sub_windows:
                final_windows.append((sub, "max_dur_split"))
        else:
            final_windows.append((segs, meth))

    # ── build output chunks ───────────────────────────────────────────
    chunks = []
    for chunk_num, (segs, meth) in enumerate(final_windows, start=1):
        if segs:
            chunks.append(_flush(segs, metadata, chunk_num, meth))

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run(
    course_ids: list[str] | None,
    threshold:  float,
    dry_run:    bool,
    suffix: str
) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED / f"segments_c3{suffix}.jsonl"

    total_chunks    = 0
    total_lectures  = 0
    ocr_used        = 0    # chunks where boundary was detected via OCR
    fallback_used   = 0    # chunks where fallback was used
    course_stats: dict[str, dict] = {}

    out_fh = None
    if not dry_run:
        out_fh = open(output_file, "w", encoding="utf-8")

    try:
        for metadata, segments in iter_lectures(OUTPUT_DIR, course_ids):
            cid   = metadata.get("course_id", "unknown")
            lnum  = metadata.get("lecture_number", 0)
            title = metadata.get("lecture_title", "")[:50]

            chunks = chunk_lecture_c3(segments, metadata, threshold)

            if not chunks:
                print(f"  [WARN] No chunks for {cid}/lec{lnum:03d}", flush=True)
                continue

            total_lectures += 1
            total_chunks   += len(chunks)

            lec_ocr      = sum(1 for c in chunks
                               if c["boundary_method"] == "ocr_jaccard")
            lec_fallback = sum(1 for c in chunks
                               if "fallback" in c["boundary_method"])
            ocr_used     += lec_ocr
            fallback_used += lec_fallback

            if cid not in course_stats:
                course_stats[cid] = {
                    "lectures": 0, "chunks": 0,
                    "total_dur": 0.0, "total_words": 0,
                    "ocr_boundaries": 0, "fallback_boundaries": 0,
                }
            course_stats[cid]["lectures"]           += 1
            course_stats[cid]["chunks"]             += len(chunks)
            course_stats[cid]["total_dur"]          += sum(
                c["duration_sec"] for c in chunks)
            course_stats[cid]["total_words"]        += sum(
                c["word_count"] for c in chunks)
            course_stats[cid]["ocr_boundaries"]     += lec_ocr
            course_stats[cid]["fallback_boundaries"] += lec_fallback

            if not dry_run:
                for chunk in chunks:
                    out_fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            print(f"  {cid}/lec{lnum:03d}  '{title}'  "
                  f"→ {len(chunks)} chunks "
                  f"(ocr={lec_ocr}, fallback={lec_fallback})",
                  flush=True)

    finally:
        if out_fh:
            out_fh.close()

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"C3 CHUNKING SUMMARY  (threshold={threshold})")
    print("-" * 72)
    print(f"{'COURSE':<12} {'LECS':>5} {'CHUNKS':>7} "
          f"{'AVG_DUR':>8} {'AVG_WDS':>8} {'OCR%':>6} {'FALL%':>6}")
    print("-" * 72)
    for cid, st in sorted(course_stats.items()):
        n  = st["chunks"] or 1
        ad = st["total_dur"]   / n
        aw = st["total_words"] / n
        op = st["ocr_boundaries"]     / n * 100
        fp = st["fallback_boundaries"] / n * 100
        print(f"{cid:<12} {st['lectures']:>5} {st['chunks']:>7} "
              f"{ad:>7.1f}s {aw:>8.1f} {op:>5.1f}% {fp:>5.1f}%")
    print("-" * 72)
    print(f"{'TOTAL':<12} {total_lectures:>5} {total_chunks:>7}  "
          f"OCR boundaries: {ocr_used:,}  "
          f"Fallback: {fallback_used:,}")
    print("=" * 72)

    if not dry_run:
        print(f"\n  Output → {output_file}")
        print(f"  Total segments written: {total_chunks:,}\n")
    else:
        print(f"\n  [DRY RUN] Would produce ~{total_chunks:,} segments.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="C3 chunker: slide-boundary segmentation via OCR Jaccard"
    )
    parser.add_argument("--course",    type=str,   default=None,
                        help="Process only this course id (e.g. dbms).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Jaccard similarity threshold for slide change "
                             f"(default {DEFAULT_THRESHOLD}). "
                             f"Lower = more sensitive to changes.")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print stats without writing output file.")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    course_ids = [args.course] if args.course else None

    print(f"C3 Chunker — threshold={args.threshold} | "
          f"min_sec={MIN_CHUNK_SEC} | max_sec={MAX_CHUNK_SEC} | "
          f"dry_run={args.dry_run}")

    run(course_ids, args.threshold, args.dry_run, args.suffix)


if __name__ == "__main__":
    main()
