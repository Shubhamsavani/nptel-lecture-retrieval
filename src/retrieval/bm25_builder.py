"""
bm25_builder.py  —  Phase 2A: Sparse (BM25) Index Builder
==========================================================
Builds a BM25 keyword index for each chunking strategy and saves it
as a pickle file so the retriever can load it instantly.

Why BM25 alongside dense retrieval:
  Dense retrieval excels at semantic/paraphrase queries but can miss
  exact technical terms. BM25 excels at exact keyword matches.
  Example: "Dijkstra" or "BST insertion" are exact terms that BGE
  may not rank highly if the passage paraphrases the concept.
  Combining both (Reciprocal Rank Fusion in retriever.py) consistently
  outperforms either approach alone.

BM25 text representation:
  Concatenation of cleaned OCR + transcript, lowercased, stopwords
  removed. Technical terms (algorithm names, data structures) are
  preserved — they are the exact terms users search for.

Usage
-----
    python bm25_builder.py                    # build all strategies
    python bm25_builder.py --strategy c1      # one strategy only
    python bm25_builder.py --dry-run          # print plan only

Output
------
    data/indexes/
        bm25_c1.pkl
        bm25_c2.pkl
        bm25_c3.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
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

STRATEGIES = {
    "c1": "segments_c1.jsonl",
    "c2": "segments_c2.jsonl",
    "c3": "segments_c3.jsonl",
}

# ── stopwords — keep technical terms, remove common words ────────────────────
# Deliberately small set: we want to keep CS terms like "for", "set", "list"
# which are common English words but important CS vocabulary.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of",
    "is", "it", "as", "be", "by", "we", "so", "do", "if", "he", "she",
    "this", "that", "with", "from", "are", "was", "were", "has", "have",
    "had", "not", "also", "will", "can", "its", "our", "their", "what",
    "which", "when", "there", "then", "they", "them", "been", "more",
    "into", "than", "just", "some", "would", "about", "because", "now",
    "very", "here", "like", "okay", "right", "yeah", "uh", "um",
}


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def tokenise(text: str) -> list[str]:
    """
    Lowercases, removes punctuation, splits into tokens, removes stopwords.
    Preserves technical tokens like "O(n)", "BST", "IPv4", "Wi-Fi".
    Minimum token length: 2 chars (keeps "OS", "ML", etc.)
    """
    if not text:
        return []
    # lowercase and replace noise chars
    text = text.lower()
    text = re.sub(r'[^\w\s\-]', ' ', text)
    tokens = text.split()
    return [
        t for t in tokens
        if len(t) >= 2 and t not in _STOPWORDS
    ]


def build_bm25_text(seg: dict) -> str:
    """
    Builds the text string fed to BM25 for a segment.
    Prioritises slide text (OCR) over speech because OCR contains
    exact technical terms (algorithm names, variable names, etc.)
    that users are likely to search for verbatim.
    """
    parts = []

    # course name adds domain context for exact match on "networks", "OS", etc.
    course = seg.get("course_name", "")
    if course:
        parts.append(course)

    # OCR text — weight doubled by repeating it once
    # This is the BM25 equivalent of "boosting" slide text
    ocr = seg.get("ocr_text", "").strip()
    if ocr and not seg.get("ocr_failed", False):
        ocr_clean = ocr.replace("\n---\n", " ").replace("\n", " ").strip()
        if len(ocr_clean) > 10:
            parts.append(ocr_clean)
            parts.append(ocr_clean)   # repeat once = ×2 weight in BM25

    # transcript
    transcript = seg.get("transcript", "").strip()
    if transcript:
        parts.append(transcript)

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# BM25 index builder
# ─────────────────────────────────────────────────────────────────────────────

def build_bm25_index(
    strategy:   str,
    jsonl_path: Path,
    dry_run:    bool,
) -> None:
    """
    Builds and saves a BM25 index for one strategy.

    The saved pickle contains:
        {
            "bm25":   BM25Okapi object,
            "corpus": list of tokenised documents (parallel to .jsonl rows)
        }

    The retriever loads this and calls bm25.get_scores(query_tokens).
    """
    from rank_bm25 import BM25Okapi

    print(f"\n{'='*55}", flush=True)
    print(f"Building BM25 index for strategy: {strategy.upper()}", flush=True)
    print(f"Input: {jsonl_path}", flush=True)

    if not jsonl_path.exists():
        print(f"  [SKIP] File not found: {jsonl_path}", flush=True)
        return

    # Load segments
    segments = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    segments.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"  Loaded {len(segments):,} segments", flush=True)

    if not segments:
        print(f"  [SKIP] Empty file", flush=True)
        return

    if dry_run:
        sample_text = build_bm25_text(segments[0])
        sample_toks = tokenise(sample_text)
        print(f"  [DRY RUN] Would index {len(segments):,} segments", flush=True)
        print(f"  Sample tokens[0]: {sample_toks[:20]}", flush=True)
        return

    # Tokenise all segments
    print(f"  Tokenising {len(segments):,} segments ...", flush=True)
    t0 = time.time()
    corpus = [tokenise(build_bm25_text(seg)) for seg in segments]

    # Filter out completely empty tokenised docs (shouldn't happen, safety net)
    empty_count = sum(1 for doc in corpus if not doc)
    if empty_count:
        print(f"  [WARN] {empty_count} segments produced empty token lists",
              flush=True)

    # Build BM25 index
    bm25 = BM25Okapi(corpus)
    elapsed = time.time() - t0
    print(f"  BM25 built in {elapsed:.1f}s", flush=True)

    # Save
    INDEXES.mkdir(parents=True, exist_ok=True)
    out_path = INDEXES / f"bm25_{strategy}.pkl"

    with open(out_path, "wb") as fh:
        pickle.dump({"bm25": bm25, "corpus": corpus}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved BM25 index → {out_path}  ({size_mb:.1f} MB)", flush=True)

    # Vocabulary stats
    vocab_size = len(bm25.idf)
    avg_doc_len = sum(len(d) for d in corpus) / len(corpus)
    print(f"  Vocab size: {vocab_size:,} | Avg doc length: {avg_doc_len:.1f} tokens",
          flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BM25 sparse indexes for C1/C2/C3 segment files"
    )
    parser.add_argument("--strategy", type=str, default=None,
                        choices=list(STRATEGIES.keys()),
                        help="Build only this strategy's index.")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print plan without building or saving.")
    args = parser.parse_args()

    strategies = (
        {args.strategy: STRATEGIES[args.strategy]}
        if args.strategy
        else STRATEGIES
    )

    for strategy_id, fname in strategies.items():
        jsonl_path = PROCESSED / fname
        build_bm25_index(strategy_id, jsonl_path, args.dry_run)

    if not args.dry_run:
        print("\nAll BM25 indexes built. Ready for retriever.py\n")


if __name__ == "__main__":
    main()
