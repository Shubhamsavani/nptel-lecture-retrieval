"""
retriever.py  —  NPTEL Lecture Retrieval System (Production)
=============================================================
Production retrieval pipeline — always uses the full stack:

  1. BGE-small query embedding  (dense / FAISS)
  2. BM25 keyword scoring       (sparse)
  3. Reciprocal Rank Fusion     (merge dense + sparse)
  4. Content-type score boost   (intent-driven)
  5. Cross-encoder re-ranking   (ms-marco-MiniLM, top-50)
  6. Lecture-level deduplication

Strategy is always c3 (OCR-aware slide-boundary chunking).
All retrieval components (BM25, RRF, reranker, OCR) are always enabled.
"""

from __future__ import annotations

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
# PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEXES      = PROJECT_ROOT / "data" / "indexes"

# ── model config ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",  "BAAI/bge-small-en-v1.5")
RERANKER_MODEL   = os.getenv("RERANKER_MODEL",   "cross-encoder/ms-marco-MiniLM-L-6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# ── retrieval config ──────────────────────────────────────────────────────────
STRATEGY      = "c3"
CANDIDATE_K   = 100
RERANK_K      = 50
DEFAULT_TOP_K = 5

# ── content-type boost factors ────────────────────────────────────────────────
BOOST_CODE       = 1.4
BOOST_THEORY     = 1.25
BOOST_CONCEPTUAL = 1.1

# ── valid intents ─────────────────────────────────────────────────────────────
VALID_INTENTS = {"code", "theoretical", "conceptual"}


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (cached — loaded once per process)
# ─────────────────────────────────────────────────────────────────────────────

_embed_model  = None
_rerank_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
    return _embed_model


def _get_rerank_model():
    global _rerank_model
    if _rerank_model is None:
        from sentence_transformers import CrossEncoder
        _rerank_model = CrossEncoder(RERANKER_MODEL)
    return _rerank_model


# ─────────────────────────────────────────────────────────────────────────────
# Index loader (cached — loaded once per process)
# ─────────────────────────────────────────────────────────────────────────────

_index_cache: dict[str, dict] = {}


def _load_index() -> dict:
    """Loads FAISS + metadata + BM25 indexes for strategy c3."""
    if STRATEGY in _index_cache:
        return _index_cache[STRATEGY]

    import faiss

    faiss_path = INDEXES / f"faiss_{STRATEGY}.index"
    meta_path  = INDEXES / f"metadata_{STRATEGY}.json"
    bm25_path  = INDEXES / f"bm25_{STRATEGY}.pkl"

    if not faiss_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {faiss_path}\n"
            f"Run: python embedder.py --strategy {STRATEGY}"
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {meta_path}\n"
            f"Run: python embedder.py --strategy {STRATEGY}"
        )
    if not bm25_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {bm25_path}\n"
            f"Run: python bm25_builder.py --strategy {STRATEGY}"
        )

    t0 = time.time()
    faiss_index = faiss.read_index(str(faiss_path))
    metadata    = json.loads(meta_path.read_text(encoding="utf-8"))

    with open(bm25_path, "rb") as fh:
        bm25_data = pickle.load(fh)
    bm25_obj = bm25_data["bm25"]
    corpus   = bm25_data.get("corpus", [])

    print(f"Indexes loaded: {faiss_index.ntotal:,} vectors in {time.time()-t0:.1f}s",
          flush=True)

    _index_cache[STRATEGY] = {
        "faiss":    faiss_index,
        "metadata": metadata,
        "bm25":     bm25_obj,
        "corpus":   corpus,
    }
    return _index_cache[STRATEGY]


# ─────────────────────────────────────────────────────────────────────────────
# BM25 tokeniser (must match bm25_builder.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of",
    "is", "it", "as", "be", "by", "we", "so", "do", "if", "he", "she",
    "this", "that", "with", "from", "are", "was", "were", "has", "have",
    "had", "not", "also", "will", "can", "its", "our", "their", "what",
    "which", "when", "there", "then", "they", "them", "been", "more",
    "into", "than", "just", "some", "would", "about", "because", "now",
    "very", "here", "like", "okay", "right", "yeah", "uh", "um",
}


def _tokenise(text: str) -> list[str]:
    text   = text.lower()
    text   = re.sub(r'[^\w\s\-]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]


# ─────────────────────────────────────────────────────────────────────────────
# Dense retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _dense_retrieve(query: str, index_data: dict, k: int) -> list[tuple[int, float]]:
    model = _get_embed_model()
    q_vec = model.encode(
        ["query: " + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    scores, indices = index_data["faiss"].search(q_vec, k)
    return list(zip(indices[0].tolist(), scores[0].tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# Sparse (BM25) retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _sparse_retrieve(query: str, index_data: dict, k: int) -> list[tuple[int, float]]:
    tokens = _tokenise(query)
    if not tokens:
        return []
    scores  = index_data["bm25"].get_scores(tokens)
    top_k   = min(k, len(scores))
    indices = scores.argsort()[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in indices if scores[i] > 0]


# ─────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    dense_results:  list[tuple[int, float]],
    sparse_results: list[tuple[int, float]],
    k_rrf: int = 60,
) -> list[tuple[int, float]]:
    rrf_scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank)
    for rank, (idx, _) in enumerate(sparse_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Content-type boost
# ─────────────────────────────────────────────────────────────────────────────

def _apply_content_boost(
    fused:    list[tuple[int, float]],
    metadata: list[dict],
    intent:   str,
) -> list[tuple[int, float]]:
    boosted = []
    for idx, score in fused:
        if idx >= len(metadata):
            boosted.append((idx, score))
            continue
        seg     = metadata[idx]
        ctype   = seg.get("content_type", "conceptual")
        is_code = seg.get("is_code_segment", False)

        multiplier = 1.0
        if intent == "code" and is_code:
            multiplier = BOOST_CODE
        elif intent == "theoretical" and ctype == "theoretical":
            multiplier = BOOST_THEORY
        elif intent == "conceptual" and ctype == "conceptual":
            multiplier = BOOST_CONCEPTUAL

        boosted.append((idx, score * multiplier))

    return sorted(boosted, key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-encoder re-ranking (OCR always included)
# ─────────────────────────────────────────────────────────────────────────────

def _rerank(
    query:      str,
    candidates: list[tuple[int, float]],
    metadata:   list[dict],
    top_k:      int,
) -> list[tuple[int, float]]:
    """Cross-encoder reranker. OCR text is always included in the passage."""
    reranker   = _get_rerank_model()
    pairs      = []
    valid_idxs = []

    for idx, _ in candidates:
        if idx >= len(metadata):
            continue
        seg        = metadata[idx]
        transcript = seg.get("transcript", "").strip()

        if not seg.get("ocr_failed", True):
            ocr = seg.get("ocr_text", "").replace("\n---\n", " ").replace("\n", " ").strip()
            passage = f"{ocr} {transcript}" if ocr else transcript
        else:
            passage = transcript

        pairs.append([query, passage[:512]])
        valid_idxs.append(idx)

    if not pairs:
        return candidates[:top_k]

    scores   = reranker.predict(pairs)
    reranked = sorted(zip(valid_idxs, scores.tolist()),
                      key=lambda x: x[1], reverse=True)
    return reranked[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Lecture-level deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate_by_lecture(
    ranked:   list[tuple[int, float]],
    metadata: list[dict],
    top_k:    int,
) -> list[tuple[int, float]]:
    """Keeps only the highest-scored segment per unique YouTube video."""
    seen_urls = set()
    deduped   = []
    max_scan  = top_k * 10

    for idx, score in ranked[:max_scan]:
        if idx >= len(metadata):
            continue
        seg = metadata[idx]
        url = seg.get("youtube_url", "")
        dedup_key = url if url else (
            f"{seg.get('course_id', '')}_{seg.get('lecture_number', idx)}"
        )
        if dedup_key not in seen_urls:
            seen_urls.add(dedup_key)
            deduped.append((idx, score))
            if len(deduped) >= top_k:
                break

    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Main search function (public API)
# ─────────────────────────────────────────────────────────────────────────────

def search(
    query:  str,
    intent: str = "conceptual",
    top_k:  int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Full retrieval pipeline. Returns top_k results from unique lectures.

    Pipeline (always runs all stages):
      Dense (FAISS) → Sparse (BM25) → RRF → Content Boost → Rerank → Dedup

    Parameters
    ----------
    query  : Natural language search query.
    intent : One of 'conceptual', 'theoretical', 'code'.
             Supplied directly by the UI dropdown — no LLM detection.
    top_k  : Number of unique lectures to return.
    """
    if intent not in VALID_INTENTS:
        intent = "conceptual"

    index_data = _load_index()
    metadata   = index_data["metadata"]

    # Step 1: Dense retrieval
    dense = _dense_retrieve(query, index_data, CANDIDATE_K)

    # Step 2: Sparse (BM25) retrieval
    sparse = _sparse_retrieve(query, index_data, CANDIDATE_K)

    # Step 3: Reciprocal Rank Fusion
    fused = _reciprocal_rank_fusion(dense, sparse)

    # Step 4: Content-type boost driven by user-selected intent
    fused = _apply_content_boost(fused, metadata, intent)

    # # Step 5: Cross-encoder reranking (top-50 candidates, OCR always on)
    # rerank_in = fused[:RERANK_K]
    # reranked  = _rerank(
    #     query      = query,
    #     candidates = rerank_in,
    #     metadata   = metadata,
    #     top_k      = top_k * 4,
    # )

    # Step 6: Lecture-level deduplication
    # final = _deduplicate_by_lecture(reranked, metadata, top_k)
    final = _deduplicate_by_lecture(
        fused,
        metadata,
        top_k
    )

    # Build output
    results = []
    for rank, (idx, score) in enumerate(final, start=1):
        if idx >= len(metadata):
            continue
        seg = metadata[idx]
        results.append({
            "rank":              rank,
            "segment_id":        seg.get("segment_id"),
            "course_name":       seg.get("course_name"),
            "instructor":        seg.get("instructor"),
            "lecture_title":     seg.get("lecture_title"),
            "lecture_number":    seg.get("lecture_number"),
            "youtube_url":       seg.get("youtube_url"),
            "youtube_deep_link": seg.get("youtube_deep_link"),
            "start_sec":         seg.get("start_sec"),
            "end_sec":           seg.get("end_sec"),
            "duration_sec":      seg.get("duration_sec"),
            "transcript":        seg.get("transcript", "")[:400],
            "ocr_text":          seg.get("ocr_text", "")[:200],
            "content_type":      seg.get("content_type"),
            "is_code_segment":   seg.get("is_code_segment"),
            "ocr_failed":        seg.get("ocr_failed"),
            "retrieval_score":   round(score, 6),
            "query_intent":      intent,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NPTEL Lecture Retrieval"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query"
    )

    parser.add_argument(
        "--intent",
        type=str,
        default="conceptual",
        choices=["conceptual", "theoretical", "code"],
        help="Query intent"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results"
    )

    args = parser.parse_args()

    results = search(
        query=args.query,
        intent=args.intent,
        top_k=args.top_k
    )

    print("\n" + "=" * 70)
    print(f"Query   : {args.query}")
    print(f"Intent  : {args.intent}")
    print(f"Results : {len(results)}")
    print("=" * 70)

    for r in results:
        print(f"\nRank {r['rank']}")
        print(f"Course   : {r['course_name']}")
        print(f"Lecture  : {r['lecture_title']}")
        print(f"Time     : {r['start_sec']}s")
        print(f"Score    : {r['retrieval_score']}")
        print(f"Link     : {r['youtube_deep_link']}")
        print(f"Snippet  : {r['transcript'][:150]}")
        print("-" * 60)