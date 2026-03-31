"""
embedder.py  —  Phase 2A: Dense Index Builder
==============================================
Reads segments_c1/c2/c3.jsonl and builds one FAISS index per strategy.
Also saves a companion metadata JSON so the retriever can map vector
row numbers back to segment records.

Model: BAAI/bge-large-en-v1.5
  - 1024-dim embeddings
  - Requires "passage: " prefix on documents
  - Requires "query: "   prefix on queries  (used in retriever.py)
  - Best open-source model for passage retrieval as of 2025

Index type: FAISS IndexFlatIP (inner product on L2-normalised vectors)
  - Equivalent to cosine similarity after normalisation
  - Exact search (no approximation) — fine for < 50k segments
  - Loads fully into RAM (~16MB per index at 1024-dim float32)

Usage
-----
    python embedder.py                        # embed all strategies
    python embedder.py --strategy c1          # one strategy only
    python embedder.py --strategy c1 --batch 32  # smaller batch for low VRAM
    python embedder.py --dry-run              # print plan, no embedding

Output
------
    data/indexes/
        faiss_c1.index   + metadata_c1.json
        faiss_c2.index   + metadata_c2.json
        faiss_c3.index   + metadata_c3.json

Each metadata_cN.json is a list where index i contains the full segment
dict for the vector at row i in the FAISS index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
PROCESSED    = PROJECT_ROOT / "data" / "processed"
INDEXES      = PROJECT_ROOT / "data" / "indexes"

# ── model config ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDING_DEVICE  = os.getenv("EMBEDDING_DEVICE", "cuda")   # "cpu" fallback
DEFAULT_BATCH     = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# ── strategies to build ───────────────────────────────────────────────────────
# Maps strategy id → input jsonl filename
STRATEGIES = {
    "c1": "segments_c1.jsonl",
    "c2": "segments_c2.jsonl",
    "c3": "segments_c3.jsonl",
    "c2_w150": "segments_c2_w150.jsonl",
    "c2_w250": "segments_c2_w250.jsonl",
    "c3_t025": "segments_c3_t025.jsonl",
    "c3_t040": "segments_c3_t040.jsonl"
}


# ─────────────────────────────────────────────────────────────────────────────
# Text representation builder
# ─────────────────────────────────────────────────────────────────────────────

def build_passage_text(seg: dict) -> str:
    """
    Constructs the passage text that gets embedded for each segment.

    The structured prefix format serves two purposes:
      1. Gives the model domain context (course name, content type)
      2. Lets the model implicitly learn different weights for OCR vs speech
         — you do NOT manually tune a weight; the prefix structure does it

    The "passage: " prefix is required by BGE-large for document encoding.
    The corresponding "query: " prefix is applied in retriever.py.
    """
    parts = ["passage:"]

    course = seg.get("course_name", "")
    if course:
        parts.append(f"[COURSE]: {course}")

    lecture = seg.get("lecture_title", "")
    if lecture:
        parts.append(f"[LECTURE]: {lecture}")

    ctype = seg.get("content_type", "")
    if ctype:
        parts.append(f"[TYPE]: {ctype}")

    ocr = seg.get("ocr_text", "").strip()
    if ocr and not seg.get("ocr_failed", False):
        # Clean OCR text: remove separator lines added by chunkers
        ocr_clean = ocr.replace("\n---\n", " ").replace("\n", " ").strip()
        if len(ocr_clean) > 10:   # ignore trivially short OCR
            parts.append(f"[SLIDE]: {ocr_clean}")

    transcript = seg.get("transcript", "").strip()
    if transcript:
        parts.append(f"[SPEECH]: {transcript}")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_segments(jsonl_path: Path) -> list[dict]:
    """Loads all segment records from a .jsonl file."""
    segments = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    segments.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  [WARN] Skipping malformed line: {e}", flush=True)
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Index builder
# ─────────────────────────────────────────────────────────────────────────────

def build_index(
    strategy:  str,
    jsonl_path: Path,
    model,
    batch_size: int,
    dry_run:   bool,
) -> None:
    """
    Embeds all segments from jsonl_path and saves:
        data/indexes/faiss_{strategy}.index
        data/indexes/metadata_{strategy}.json
    """
    import numpy as np
    import faiss

    print(f"\n{'='*55}", flush=True)
    print(f"Building index for strategy: {strategy.upper()}", flush=True)
    print(f"Input: {jsonl_path}", flush=True)

    if not jsonl_path.exists():
        print(f"  [SKIP] File not found: {jsonl_path}", flush=True)
        return

    segments = load_segments(jsonl_path)
    if not segments:
        print(f"  [SKIP] No segments loaded from {jsonl_path}", flush=True)
        return

    print(f"  Loaded {len(segments):,} segments", flush=True)

    # Build passage texts
    texts = [build_passage_text(seg) for seg in segments]

    if dry_run:
        print(f"  [DRY RUN] Would embed {len(texts):,} passages", flush=True)
        print(f"  Sample text[0]:\n    {texts[0][:200]}...", flush=True)
        return

    # ── embed ─────────────────────────────────────────────────────────────
    print(f"  Embedding {len(texts):,} passages "
          f"(batch={batch_size}, device={EMBEDDING_DEVICE}) ...", flush=True)

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,    # L2 normalise → cosine via inner product
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    print(f"  Embedding done in {elapsed:.1f}s "
          f"({len(texts)/elapsed:.0f} passages/sec)", flush=True)

    # embeddings shape: (N, 1024)
    embeddings = embeddings.astype("float32")
    dim        = embeddings.shape[1]

    # ── build FAISS index ─────────────────────────────────────────────────
    index = faiss.IndexFlatIP(dim)    # inner product on normalised vecs = cosine
    index.add(embeddings)
    print(f"  FAISS index built: {index.ntotal:,} vectors, dim={dim}", flush=True)

    # ── save ──────────────────────────────────────────────────────────────
    INDEXES.mkdir(parents=True, exist_ok=True)

    index_path = INDEXES / f"faiss_{strategy}.index"
    meta_path  = INDEXES / f"metadata_{strategy}.json"

    faiss.write_index(index, str(index_path))
    print(f"  Saved FAISS index → {index_path}", flush=True)

    # metadata: keep only the fields needed for display + deep link
    # storing full segments keeps retriever simple (no secondary lookup)
    meta_fields = [
        "segment_id", "course_id", "course_name", "instructor", "institute",
        "lecture_title", "lecture_number", "youtube_url", "youtube_deep_link",
        "start_sec", "end_sec", "duration_sec",
        "transcript", "ocr_text", "ocr_confidence", "ocr_failed",
        "is_code_segment", "content_type", "chunking_strategy",
        "word_count", "boundary_method",
    ]
    slim_meta = [
        {k: seg.get(k) for k in meta_fields}
        for seg in segments
    ]
    meta_path.write_text(
        json.dumps(slim_meta, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"  Saved metadata     → {meta_path}", flush=True)
    print(f"  Index size on disk : {index_path.stat().st_size / 1e6:.1f} MB",
          flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS dense indexes for C1/C2/C3 segment files"
    )
    parser.add_argument("--strategy", type=str, default=None,
                        choices=list(STRATEGIES.keys()),
                        help="Build only this strategy's index (c1, c2, or c3).")
    parser.add_argument("--batch",    type=int, default=DEFAULT_BATCH,
                        help=f"Embedding batch size (default {DEFAULT_BATCH}). "
                             "Reduce to 32 or 16 if you get CUDA OOM.")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print plan without embedding or saving.")
    parser.add_argument("--cpu",      action="store_true",
                        help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    device = "cpu" if args.cpu else EMBEDDING_DEVICE

    strategies = (
        {args.strategy: STRATEGIES[args.strategy]}
        if args.strategy
        else STRATEGIES
    )

    if args.dry_run:
        print(f"[DRY RUN] Would embed strategies: {list(strategies.keys())}")
        for sid, fname in strategies.items():
            path = PROCESSED / fname
            exists = path.exists()
            print(f"  {sid}: {path} — {'EXISTS' if exists else 'MISSING'}")
        return

    # ── load model once ───────────────────────────────────────────────────
    print(f"Loading embedding model: {EMBEDDING_MODEL} on {device} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    print(f"Model loaded. Embedding dim = {model.get_sentence_embedding_dimension()}")

    # ── build each index ──────────────────────────────────────────────────
    for strategy_id, fname in strategies.items():
        jsonl_path = PROCESSED / fname
        build_index(strategy_id, jsonl_path, model, args.batch, args.dry_run)

    print("\nAll indexes built. Ready for retriever.py\n")


if __name__ == "__main__":
    main()
