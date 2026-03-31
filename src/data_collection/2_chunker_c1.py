"""
chunker_c1.py  —  Strategy C1: Fixed 30-Second Window
======================================================
Baseline chunking strategy. Walks through segments of a lecture and
groups them into non-overlapping 30-second windows. Chunk boundaries
are placed at the nearest segment boundary after 30 seconds have elapsed.

This is the simplest possible approach and serves as the baseline
against which C2 and C3 are compared in the thesis ablation study.

Known weakness (which C2 and C3 address):
  - Cuts mid-sentence when a segment straddles the 30s boundary
  - Ignores semantic structure entirely — two different topics may
    land in the same chunk if they happen to fall within 30 seconds

Usage
-----
    python chunker_c1.py                    # all courses
    python chunker_c1.py --course dbms      # one course
    python chunker_c1.py --course dbms --window 45   # custom window size
    python chunker_c1.py --dry-run          # stats only, no file written

Output
------
    data/processed/segments_c1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
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

# ── add src to path so segment_utils is importable ───────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_utils import (
    build_segment_id,
    build_deep_link,
    classify_chunk,
    ocr_confidence,
    is_ocr_failed,
    iter_lectures,
)

# ── chunking config ───────────────────────────────────────────────────────────
DEFAULT_WINDOW_SEC = 30   # seconds per chunk
STRATEGY_ID        = "c1"
OUTPUT_FILE        = PROCESSED / "segments_c1.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# C1 chunking logic
# ─────────────────────────────────────────────────────────────────────────────

def chunk_lecture_c1(
    segments:    list[dict],
    metadata:    dict,
    window_sec:  float = DEFAULT_WINDOW_SEC,
) -> list[dict]:
    """
    Splits a lecture's segments into fixed-duration windows.

    Args:
        segments   — normalised list from load_multimodal()
        metadata   — dict from metadata.json
        window_sec — target window duration in seconds

    Returns:
        List of chunk dicts ready to write to segments_c1.jsonl
    """
    if not segments:
        return []

    course_id   = metadata.get("course_id", "unknown")
    video_id    = metadata.get("video_id", "")
    lec_num     = metadata.get("lecture_number", 0)
    lec_title   = metadata.get("lecture_title", metadata.get("lecture_folder", ""))
    course_name = metadata.get("course_name", "")
    instructor  = metadata.get("instructor", "")
    institute   = metadata.get("institute", "")

    chunks       = []
    chunk_num    = 0
    window_segs  = []          # segments accumulated in the current window
    window_start = segments[0]["start"]

    def _flush(window_segs: list[dict], chunk_num: int) -> dict:
        """Converts accumulated segments into one chunk record."""
        transcript = " ".join(s["text"] for s in window_segs).strip()

        # OCR: concatenate unique non-empty OCR texts in this window
        # deduplicate consecutive identical OCR blocks (same slide repeated)
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
            "segment_id":       build_segment_id(course_id, lec_num,
                                                 STRATEGY_ID, chunk_num),
            "course_id":        course_id,
            "course_name":      course_name,
            "instructor":       instructor,
            "institute":        institute,
            "lecture_title":    lec_title,
            "lecture_number":   lec_num,
            "youtube_url":      metadata.get("youtube_url", ""),
            "youtube_deep_link": build_deep_link(video_id, start_sec),
            "start_sec":        round(start_sec, 2),
            "end_sec":          round(end_sec, 2),
            "duration_sec":     dur_sec,
            "transcript":       transcript,
            "ocr_text":         ocr_text,
            "ocr_confidence":   conf,
            "ocr_failed":       failed,
            "is_code_segment":  is_code,
            "content_type":     ctype,
            "chunking_strategy": "C1_fixed",
            "word_count":       wc,
            "source_pipeline":  metadata.get("source_pipeline", "whisper_ocr"),
        }

    for seg in segments:
        window_segs.append(seg)
        elapsed = seg["end"] - window_start

        if elapsed >= window_sec:
            chunk_num += 1
            chunks.append(_flush(window_segs, chunk_num))
            window_segs  = []
            # next window starts at the end of the segment we just closed
            window_start = seg["end"]

    # flush the final (possibly short) window
    if window_segs:
        chunk_num += 1
        chunks.append(_flush(window_segs, chunk_num))

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run(course_ids: list[str] | None, window_sec: float, dry_run: bool) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    total_chunks    = 0
    total_lectures  = 0
    course_stats: dict[str, dict] = {}

    # open output file for writing (overwrite if exists)
    out_fh = None
    if not dry_run:
        out_fh = open(OUTPUT_FILE, "w", encoding="utf-8")

    try:
        for metadata, segments in iter_lectures(OUTPUT_DIR, course_ids):
            cid   = metadata.get("course_id", "unknown")
            lnum  = metadata.get("lecture_number", 0)
            title = metadata.get("lecture_title", "")[:50]

            chunks = chunk_lecture_c1(segments, metadata, window_sec)

            if not chunks:
                print(f"  [WARN] No chunks for {cid}/lec{lnum:03d}", flush=True)
                continue

            total_lectures += 1
            total_chunks   += len(chunks)

            if cid not in course_stats:
                course_stats[cid] = {"lectures": 0, "chunks": 0}
            course_stats[cid]["lectures"] += 1
            course_stats[cid]["chunks"]   += len(chunks)

            if not dry_run:
                for chunk in chunks:
                    out_fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            print(f"  {cid}/lec{lnum:03d}  '{title}'  → {len(chunks)} chunks",
                  flush=True)

    finally:
        if out_fh:
            out_fh.close()

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"C1 CHUNKING SUMMARY  (window={window_sec}s)")
    print("-" * 60)
    print(f"{'COURSE':<12} {'LECTURES':>9} {'CHUNKS':>8} {'AVG/LEC':>9}")
    print("-" * 60)
    for cid, st in sorted(course_stats.items()):
        avg = st["chunks"] / st["lectures"] if st["lectures"] else 0
        print(f"{cid:<12} {st['lectures']:>9} {st['chunks']:>8} {avg:>9.1f}")
    print("-" * 60)
    print(f"{'TOTAL':<12} {total_lectures:>9} {total_chunks:>8}")
    print("=" * 60)

    if not dry_run:
        print(f"\n  Output → {OUTPUT_FILE}")
        print(f"  Total segments written: {total_chunks:,}\n")
    else:
        print(f"\n  [DRY RUN] Nothing written. Would produce ~{total_chunks:,} segments.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="C1 chunker: fixed time-window segmentation"
    )
    parser.add_argument("--course",  type=str,   default=None,
                        help="Process only this course id (e.g. dbms).")
    parser.add_argument("--window",  type=float, default=DEFAULT_WINDOW_SEC,
                        help=f"Window size in seconds (default {DEFAULT_WINDOW_SEC}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing output file.")
    args = parser.parse_args()

    course_ids = [args.course] if args.course else None

    print(f"C1 Chunker — window={args.window}s | "
          f"dry_run={args.dry_run} | output_dir={OUTPUT_DIR}")

    run(course_ids, args.window, args.dry_run)


if __name__ == "__main__":
    main()
