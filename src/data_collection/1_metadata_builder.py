"""
metadata_builder.py  —  Pre-chunking metadata enrichment
=========================================================
Scans the Pipeline B output folder and writes a metadata.json
alongside every multimodal.json that does not already have one.

What it does per lecture
------------------------
1. Reads courses.json to get course-level info (name, instructor, etc.)
2. Fetches the YouTube playlist for that course using yt-dlp --flat-playlist
3. Matches the lecture folder name to a video title in the playlist
4. Writes metadata.json with video_id, youtube_url, lecture_number, etc.

Manual override
---------------
If a folder name does not match any playlist title (e.g. the filename was
truncated or sanitised differently), the script logs it as UNMATCHED and
you can add it to MANUAL_OVERRIDES below.

MANUAL_OVERRIDES format:
    key   = exact folder name (Path.name of the lecture directory)
    value = {"video_id": "XXXXXXXXXXX", "lecture_number": 3}

Usage
-----
    python metadata_builder.py                  # process all courses
    python metadata_builder.py --course dbms    # one course only
    python metadata_builder.py --dry-run        # print plan, no writes

Output
------
    output/<course>/<lecture>/metadata.json

Requirements
------------
    yt-dlp must be installed and cookies.txt must be present (for playlist fetch)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import random
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
CONFIGS_DIR  = PROJECT_ROOT / "configs"
COURSES_JSON = CONFIGS_DIR  / "courses.json"
COOKIE_PATH  = PROJECT_ROOT / os.getenv("YTDLP_COOKIE_FILE", "cookies.txt")

# Pipeline B output root — change if your output folder is elsewhere
# DATA_DIR = Path(os.getenv("DATA_DIR"))

OUTPUT_DIR = Path(os.getenv("DATA_DIR"))

# ── manual overrides ──────────────────────────────────────────────────────────
# Add entries here when the folder name does not match the YouTube title.
# Format:  "exact_folder_name": {"video_id": "XXXXXXXXXXX", "lecture_number": N}
MANUAL_OVERRIDES: dict[str, dict] = {
    # Example:
    # "Lecture 1 - Introduction to DBMS": {"video_id": "abc123XYZ", "lecture_number": 1},
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str, level: str = "INFO") -> None:
    icons = {"INFO": "  ", "OK": "✅", "WARN": "⚠️ ", "FAIL": "❌", "SKIP": "⏭️ "}
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {icons.get(level,'  ')} {msg}",
          flush=True)


def _load_courses(course_filter: str | None = None) -> dict[str, dict]:
    """Returns {course_id: course_dict} from courses.json."""
    if not COURSES_JSON.exists():
        sys.exit(f"❌  courses.json not found at {COURSES_JSON}")
    data = json.loads(COURSES_JSON.read_text(encoding="utf-8"))
    result = {c["id"]: c for c in data["courses"]}
    if course_filter:
        if course_filter not in result:
            sys.exit(f"❌  Course '{course_filter}' not in courses.json")
        return {course_filter: result[course_filter]}
    return result


def _fetch_playlist(playlist_url: str, course_id: str) -> list[dict]:
    """
    Fetches video list from a YouTube playlist using yt-dlp --flat-playlist.
    Returns list of {video_id, title, position} dicts.
    """
    _log(f"Fetching playlist for [{course_id}] ...")
    cookie_args = ["--cookies", str(COOKIE_PATH)] if COOKIE_PATH.exists() else []

    cmd = [
        "yt-dlp",
        *cookie_args,
        "--flat-playlist",
        "--print", "%(id)s\t%(title)s\t%(playlist_index)s",
        "--no-warnings",
        playlist_url,
    ]

    for attempt in range(1, 4):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                _log(f"Playlist fetch attempt {attempt} failed: "
                     f"{result.stderr.strip()[:120]}", "WARN")
                time.sleep(2 ** attempt + random.uniform(0, 2))
                continue

            videos = []
            for line in result.stdout.strip().splitlines():
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                vid_id, title, pos = parts[0], parts[1], parts[2]
                try:
                    pos_int = int(pos) if str(pos).isdigit() else len(videos) + 1
                except (ValueError, TypeError):
                    pos_int = len(videos) + 1
                videos.append({
                    "video_id": vid_id,
                    "title":    title,
                    "position": pos_int,
                })
            _log(f"Found {len(videos)} videos in playlist", "OK")
            return videos

        except subprocess.TimeoutExpired:
            _log(f"Playlist fetch timed out (attempt {attempt})", "WARN")
            time.sleep(5)

    _log(f"Failed to fetch playlist for {course_id}", "FAIL")
    return []


def _title_similarity(folder_name: str, video_title: str) -> float:
    """
    Word-overlap Jaccard similarity between folder name and YouTube title.
    Used when exact matching fails.
    """
    def words(s: str) -> set[str]:
        return set(
            w.lower() for w in s.replace("-", " ").replace("_", " ").split()
            if len(w) >= 3 and w.isalpha()
        )
    a, b = words(folder_name), words(video_title)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _match_folder_to_video(
    folder_name: str,
    playlist:    list[dict],
    threshold:   float = 0.35,
) -> dict | None:
    """
    Matches a lecture folder name to a YouTube playlist entry.

    Strategy (in priority order):
      1. Exact string match (case-insensitive, stripped)
      2. Folder name is a substring of video title or vice versa
      3. Highest Jaccard word-overlap score >= threshold

    Returns best-matching playlist entry dict, or None if no match found.
    """
    folder_lower = folder_name.lower().strip()

    # Pass 1 — exact
    for v in playlist:
        if v["title"].lower().strip() == folder_lower:
            return v

    # Pass 2 — substring
    for v in playlist:
        vt = v["title"].lower().strip()
        if folder_lower in vt or vt in folder_lower:
            return v

    # Pass 3 — fuzzy
    best_score = 0.0
    best_match = None
    for v in playlist:
        score = _title_similarity(folder_name, v["title"])
        if score > best_score:
            best_score = score
            best_match = v

    if best_score >= threshold:
        return best_match

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Core builder
# ─────────────────────────────────────────────────────────────────────────────

def build_metadata_for_course(
    course:  dict,
    dry_run: bool = False,
) -> dict:
    """
    Processes all lecture folders for one course.
    Returns a summary dict.
    """
    course_id  = course["id"]
    course_dir = OUTPUT_DIR / course_id.upper()

    if not course_dir.exists():
        _log(f"Course folder not found: {course_dir} — skipping", "WARN")
        return {"course_id": course_id,
                "written": 0, "skipped": 0, "unmatched": 0}

    lecture_dirs = sorted(d for d in course_dir.iterdir() if d.is_dir())
    if not lecture_dirs:
        _log(f"No lecture folders in {course_dir}", "WARN")
        return {"course_id": course_id,
                "written": 0, "skipped": 0, "unmatched": 0}

    playlist  = _fetch_playlist(course["playlist_url"], course_id)
    written   = 0
    skipped   = 0
    unmatched = 0

    for lecture_dir in lecture_dirs:
        folder_name = lecture_dir.name
        meta_path   = lecture_dir / "metadata.json"
        mm_path     = lecture_dir / "multimodal.json"

        if not mm_path.exists():
            _log(f"SKIP {folder_name} — no multimodal.json", "SKIP")
            skipped += 1
            continue

        if meta_path.exists():
            _log(f"SKIP {folder_name} — metadata.json already exists", "SKIP")
            skipped += 1
            continue

        # ── resolve video_id ──────────────────────────────────────────
        override    = MANUAL_OVERRIDES.get(folder_name)
        video_id    = None
        lec_num     = 0
        video_title = folder_name
        match_method = "unmatched"

        if override:
            video_id     = override.get("video_id")
            lec_num      = override.get("lecture_number", 0)
            match_method = "manual"
            _log(f"  Manual override: {folder_name} → {video_id}")

        elif playlist:
            match = _match_folder_to_video(folder_name, playlist)
            if match:
                video_id     = match["video_id"]
                lec_num      = match["position"]
                video_title  = match["title"]
                match_method = "playlist"
                _log(f"  Matched: '{folder_name[:45]}' → [{video_id}]")
            else:
                _log(f"  UNMATCHED: '{folder_name}' — add to MANUAL_OVERRIDES",
                     "WARN")
                unmatched += 1

        # ── build and write metadata ──────────────────────────────────
        youtube_url = (
            f"https://www.youtube.com/watch?v={video_id}" if video_id else ""
        )

        metadata = {
            "video_id":        video_id or "",
            "youtube_url":     youtube_url,
            "lecture_folder":  folder_name,
            "lecture_title":   video_title,
            "lecture_number":  lec_num,
            "course_id":       course_id,
            "course_name":     course["name"],
            "instructor":      course["instructor"],
            "institute":       course["institute"],
            "playlist_url":    course["playlist_url"],
            "ocr_enabled":     course["ocr_enabled"],
            "content_profile": course["content_profile"],
            "source_pipeline": "whisper_ocr",
            "match_method":    match_method,
            "built_at":        datetime.now().isoformat(),
        }

        if dry_run:
            _log(f"  [DRY RUN] Would write metadata for: {folder_name}")
            written += 1
            continue

        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        written += 1
        _log(f"  Written: metadata.json for '{folder_name}'", "OK")

    _log(f"Done — {written} written / {skipped} skipped / "
         f"{unmatched} unmatched", "OK")
    return {"course_id": course_id,
            "written": written, "skipped": skipped, "unmatched": unmatched}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build metadata.json for every Pipeline B lecture folder"
    )
    parser.add_argument("--course",  type=str, default=None,
                        help="Process only this course id (e.g. dbms).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without writing any files.")
    args = parser.parse_args()

    courses = _load_courses(args.course)
    _log(f"Processing {len(courses)} course(s) | dry_run={args.dry_run}")
    _log(f"Output dir : {OUTPUT_DIR}")

    if not OUTPUT_DIR.exists():
        sys.exit(
            f"Output directory not found: {OUTPUT_DIR}\n"
            f"Set PROJECT_ROOT in your .env or run from the project root."
        )

    all_summaries = []
    for course in courses.values():
        _log("=" * 55)
        _log(f"Course: {course['name']}  [{course['id']}]")
        _log("=" * 55)
        summary = build_metadata_for_course(course, dry_run=args.dry_run)
        all_summaries.append(summary)

    # ── summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print(f"{'COURSE':<12} {'WRITTEN':>8} {'SKIPPED':>8} {'UNMATCHED':>10}")
    print("-" * 52)
    for s in all_summaries:
        print(f"{s['course_id']:<12} {s['written']:>8} "
              f"{s['skipped']:>8} {s['unmatched']:>10}")
    print("=" * 52)

    total_unmatched = sum(s["unmatched"] for s in all_summaries)
    if total_unmatched > 0:
        print(f"\n  {total_unmatched} lecture(s) could not be matched.")
        print("  Add them to MANUAL_OVERRIDES in metadata_builder.py and re-run.")
        print("  Chunkers will still run but YouTube deep links will be empty.\n")
    else:
        print("\n  All lectures matched. Ready to run chunkers.\n")


if __name__ == "__main__":
    main()