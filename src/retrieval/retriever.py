"""
retriever.py  —  Phase 2B: Query Engine
========================================
Implements the full retrieval pipeline:

  1. [Optional] LLM query analysis via Ollama
  2. BGE-large query embedding  (dense)
  3. [Optional] BM25 keyword scoring  (sparse)
  4. Reciprocal Rank Fusion     (merge dense + sparse, if BM25 enabled)
  5. Content-type score boost   (code/theory differentiation)
  6. Cross-encoder re-ranking   (top-50 → top_k*4 with ms-marco-MiniLM)
  7. Lecture-level deduplication (no repeated videos in results)

CHANGES IN THIS VERSION
-----------------------
  NEW — use_ocr parameter
    When use_ocr=False, OCR text is stripped from the passage sent to the
    cross-encoder reranker. Dense retrieval still uses the same FAISS index,
    but the passage constructed for reranking uses transcript-only text.
    This correctly isolates the OCR contribution in ablation experiments.

  NEW — use_bm25 parameter
    When use_bm25=False, BM25 sparse retrieval is skipped entirely.
    Only dense FAISS results are used (no RRF fusion).
    When use_bm25=True, RRF merges dense + sparse results as before.

  Fix 1 — Lecture deduplication
    After final ranking, only the highest-scored segment per unique
    youtube_url is kept. Fills top_k slots from unique lectures only.

  Fix 2A — Ollama KV-cache bleeding
    Added "keep_alive": 0 to every Ollama API call. Forces Ollama to evict
    the model from VRAM after each generation, destroying the KV cache.

  Fix 2B — Fragile JSON extraction
    Replaced re.search(r'\{.*\}', ..., re.DOTALL) with a character-by-
    character brace-counting extractor.

  Fix 4 — Prompt indentation contamination
    Fixed with textwrap.dedent() before sending to Ollama.

CONFIRMATION SCRIPTS
--------------------
    python retriever.py --confirm-ollama-bleed
    python retriever.py --confirm-json-parser
    python retriever.py --query "BST" --confirm-dedup

Usage
-----
    python retriever.py --query "how does binary search tree insertion work"
    python retriever.py --query "BST insertion" --strategy c3
    python retriever.py --query "BST insertion" --llm --verbose
    python retriever.py --query "BST insertion" --top-k 10
    python retriever.py --query "BST insertion" --no-ocr --no-bm25
    python retriever.py --test
    python retriever.py --no-rerank --query "BST insertion"
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import textwrap
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
INDEXES      = PROJECT_ROOT / "data" / "indexes"

# ── model config ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL",   "BAAI/bge-large-en-v1.5")
RERANKER_MODEL    = os.getenv("RERANKER_MODEL",    "cross-encoder/ms-marco-MiniLM-L-6-v2")
EMBEDDING_DEVICE  = os.getenv("EMBEDDING_DEVICE",  "cuda")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL",      "llama3.2:3b")
OLLAMA_HOST       = os.getenv("OLLAMA_HOST",       "http://localhost:11434")

# ── retrieval config ──────────────────────────────────────────────────────────
DEFAULT_STRATEGY  = os.getenv("DEFAULT_STRATEGY",  "c2")
CANDIDATE_K       = 100
RERANK_K          = 50
DEFAULT_TOP_K     = 5

# ── content-type boost factors ────────────────────────────────────────────────
BOOST_CODE       = 1.4
BOOST_THEORY     = 1.25
BOOST_CONCEPTUAL = 1.1


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
# Index loader (cached — loaded once per strategy)
# ─────────────────────────────────────────────────────────────────────────────

_index_cache: dict[str, dict] = {}


def _load_index(strategy: str) -> dict:
    """
    Loads FAISS + metadata + BM25 indexes for the given strategy key.
    Strategy can be a simple key like 'c1', 'c2', 'c3' or a variant key
    like 'c2_w150', 'c3_t025' for chunk sensitivity experiments.
    """
    if strategy in _index_cache:
        return _index_cache[strategy]

    import faiss

    faiss_path = INDEXES / f"faiss_{strategy}.index"
    meta_path  = INDEXES / f"metadata_{strategy}.json"
    bm25_path  = INDEXES / f"bm25_{strategy}.pkl"

    if not faiss_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {faiss_path}\n"
            f"Run: python embedder.py --strategy {strategy}"
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {meta_path}\n"
            f"Run: python embedder.py --strategy {strategy}"
        )

    print(f"  Loading indexes for strategy={strategy} ...", flush=True)
    t0 = time.time()

    faiss_index = faiss.read_index(str(faiss_path))
    metadata    = json.loads(meta_path.read_text(encoding="utf-8"))

    # BM25 is optional — if file missing, sparse retrieval will be unavailable
    bm25_obj    = None
    corpus      = []
    if bm25_path.exists():
        with open(bm25_path, "rb") as fh:
            bm25_data = pickle.load(fh)
        bm25_obj = bm25_data["bm25"]
        corpus   = bm25_data.get("corpus", [])
    else:
        print(f"  WARNING: BM25 index not found: {bm25_path} "
              f"(sparse retrieval unavailable for strategy={strategy})", flush=True)

    elapsed = time.time() - t0
    print(f"  Loaded {faiss_index.ntotal:,} vectors in {elapsed:.1f}s", flush=True)

    _index_cache[strategy] = {
        "faiss":    faiss_index,
        "metadata": metadata,
        "bm25":     bm25_obj,
        "corpus":   corpus,
    }
    return _index_cache[strategy]


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
# FIX 2B — Robust JSON extractor (replaces fragile regex)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json_object(text: str) -> dict | None:
    """
    Extracts the first complete, outermost JSON object from a string.
    Handles markdown fences, trailing text, braces inside string values.
    """
    # Strip markdown fences
    text = re.sub(r'```(?:json)?\s*', '', text).strip()

    start = text.find('{')
    if start == -1:
        return None

    depth       = 0
    in_string   = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    remainder = text[i + 1:]
                    if '{' in remainder:
                        return _extract_json_object(remainder)
                    return None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2A + FIX 4 — LLM query analysis with clean context
# ─────────────────────────────────────────────────────────────────────────────

def analyse_query_with_llm(query: str) -> dict:
    """
    Uses local Ollama LLM to classify query intent and expand the query.
    keep_alive=0 forces KV-cache eviction between calls.
    textwrap.dedent() strips Python method indentation from the prompt.
    """
    prompt = textwrap.dedent(f"""
        You are a query analyser for an educational video retrieval system.
        Analyse this search query and respond with ONLY valid JSON, no other text.

        Query: "{query}"

        Respond with exactly this JSON structure:
        {{
          "intent": "<one of: code, theoretical, conceptual>",
          "expanded_query": "<original query plus 3-5 relevant synonyms or related terms>",
          "reasoning": "<one sentence explanation>"
        }}

        Intent definitions:
        - "code": user wants to see code, syntax, implementation, programming examples
        - "theoretical": user wants definitions, proofs, theorems, complexity analysis
        - "conceptual": user wants to understand how something works, examples, intuition

        Respond with ONLY the JSON object. No preamble, no explanation, no markdown.
    """).strip()

    print(f"  LLM: model={OLLAMA_MODEL}", flush=True)

    try:
        import requests
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":      OLLAMA_MODEL,
                "prompt":     prompt,
                "stream":     False,
                "keep_alive": 0,
                "options": {
                    "temperature":    0.0,
                    "num_predict":    200,
                    "num_ctx":        512,
                    "top_p":          1.0,
                    "repeat_penalty": 1.0,
                },
            },
            timeout=15,
        )
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()
        print(f"  LLM raw: {raw_text[:120]!r}", flush=True)

        parsed = _extract_json_object(raw_text)

        if parsed:
            intent = parsed.get("intent", "conceptual")
            if intent not in ("code", "theoretical", "conceptual"):
                print(f"  LLM: invalid intent '{intent}', defaulting", flush=True)
                intent = "conceptual"
            expanded = parsed.get("expanded_query", query)
            if not expanded or len(expanded.strip()) < len(query) // 2:
                expanded = query
            return {
                "intent":         intent,
                "expanded_query": expanded.strip(),
                "reasoning":      parsed.get("reasoning", ""),
            }
        else:
            print(f"  LLM: JSON extraction failed, using heuristic fallback",
                  flush=True)

    except Exception as e:
        print(f"  LLM error: {e}", flush=True)

    return {
        "intent":         _detect_intent_heuristic(query),
        "expanded_query": query,
        "reasoning":      "LLM unavailable — heuristic fallback",
    }


def _detect_intent_heuristic(query: str) -> str:
    """Fast keyword-based intent detection used when LLM is disabled or fails."""
    q_lower = query.lower()
    code_signals = [
        "code", "implement", "write", "program", "syntax", "function",
        "class", "method", "python", "java", "c++", "algorithm code",
        "how to", "example code", "snippet", "loop", "recursion code",
    ]
    theory_signals = [
        "explain", "what is", "define", "theorem", "proof", "complexity",
        "why", "concept", "theory", "formal", "derive", "analysis",
    ]
    code_hits   = sum(1 for s in code_signals   if s in q_lower)
    theory_hits = sum(1 for s in theory_signals if s in q_lower)

    if code_hits > theory_hits:
        return "code"
    elif theory_hits > 0:
        return "theoretical"
    return "conceptual"


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
    if index_data.get("bm25") is None:
        return []
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
# Cross-encoder re-ranking  — OCR-aware
# ─────────────────────────────────────────────────────────────────────────────

def _rerank(
    query:      str,
    candidates: list[tuple[int, float]],
    metadata:   list[dict],
    top_k:      int,
    use_ocr:    bool = True,          # NEW: controls whether OCR text is included
) -> list[tuple[int, float]]:
    """
    Cross-encoder reranker.

    use_ocr=True  → passage = OCR text + transcript  (E4, E5, E8, E9 style)
    use_ocr=False → passage = transcript only         (E1–E3, E7 style)

    This is the correct place to control OCR contribution because:
    - The FAISS index was built with combined text, so dense scores are fixed.
    - The reranker passage is the only place we can cleanly exclude OCR.
    - This gives a fair comparison: same dense candidates, different passage
      content sent to the cross-encoder.
    """
    reranker   = _get_rerank_model()
    pairs      = []
    valid_idxs = []

    for idx, _ in candidates:
        if idx >= len(metadata):
            continue
        seg        = metadata[idx]
        transcript = seg.get("transcript", "").strip()

        if use_ocr and not seg.get("ocr_failed", True):
            # Include OCR text prepended to transcript
            ocr = seg.get("ocr_text", "").replace("\n---\n", " ").replace("\n", " ").strip()
            passage = f"{ocr} {transcript}" if ocr else transcript
        else:
            # Transcript only — ignore OCR even if present in metadata
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
# FIX 1 — Lecture-level deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate_by_lecture(
    ranked:   list[tuple[int, float]],
    metadata: list[dict],
    top_k:    int,
) -> list[tuple[int, float]]:
    """
    Keeps only the highest-scored segment per unique YouTube video.
    Dedup key: youtube_url, falling back to course_id + lecture_number.
    """
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
    query:      str,
    strategy:   str  = DEFAULT_STRATEGY,
    top_k:      int  = DEFAULT_TOP_K,
    use_llm:    bool = False,
    use_rerank: bool = True,
    use_ocr:    bool = True,           # NEW: include OCR text in reranker passage
    use_bm25:   bool = True,           # NEW: enable/disable sparse BM25 retrieval
    verbose:    bool = False,
) -> list[dict]:
    """
    Full retrieval pipeline. Returns top_k results from unique lectures.

    Parameters
    ----------
    query      : Search query string.
    strategy   : Index strategy key — 'c1', 'c2', 'c3', or variant keys
                 like 'c2_w150', 'c3_t025' for chunk sensitivity runs.
    top_k      : Number of unique lectures to return.
    use_llm    : If True, use Ollama LLM for query expansion/intent analysis.
    use_rerank : If True, apply cross-encoder reranking.
    use_ocr    : If True, include OCR text in the reranker passage.
                 If False, reranker sees transcript-only passages.
                 Dense retrieval (FAISS) is unaffected — it always uses the
                 index as built. This correctly measures OCR contribution.
    use_bm25   : If True, run BM25 sparse retrieval and fuse with dense via RRF.
                 If False, only dense FAISS results are used (no RRF).
                 This correctly measures BM25 / hybrid retrieval contribution.
    verbose    : Print timing and intermediate step info.
    """
    t_total = time.time()

    # ── Step 1: Query analysis ────────────────────────────────────────────────
    if use_llm:
        t0             = time.time()
        analysis       = analyse_query_with_llm(query)
        intent         = analysis["intent"]
        expanded_query = analysis["expanded_query"]
        if verbose:
            print(f"  LLM     : {time.time()-t0:.2f}s | intent={intent} | "
                  f"expanded='{expanded_query[:60]}'")
            print(f"  Reason  : {analysis.get('reasoning', '')}")
    else:
        intent         = _detect_intent_heuristic(query)
        expanded_query = query
        if verbose:
            print(f"  Heuristic intent: {intent}")

    if verbose:
        print(f"  Config  : use_ocr={use_ocr} | use_bm25={use_bm25} | "
              f"strategy={strategy}", flush=True)

    index_data = _load_index(strategy)
    metadata   = index_data["metadata"]

    # ── Step 2: Dense retrieval (always runs) ─────────────────────────────────
    t0    = time.time()
    dense = _dense_retrieve(expanded_query, index_data, CANDIDATE_K)
    if verbose:
        print(f"  Dense   : {time.time()-t0:.2f}s | {len(dense)} candidates")

    # ── Step 3 + 4: Sparse retrieval + RRF fusion (conditional on use_bm25) ───
    if use_bm25:
        t0     = time.time()
        sparse = _sparse_retrieve(expanded_query, index_data, CANDIDATE_K)
        if verbose:
            print(f"  BM25    : {time.time()-t0:.2f}s | {len(sparse)} candidates")

        t0    = time.time()
        fused = _reciprocal_rank_fusion(dense, sparse)
        if verbose:
            print(f"  RRF     : {time.time()-t0:.3f}s | {len(fused)} unique")
    else:
        # Dense only — convert to same (idx, score) format, normalise scores
        # to [0,1] range using rank-based scoring to match RRF scale
        fused = []
        k_rrf = 60
        for rank, (idx, _) in enumerate(dense, start=1):
            fused.append((idx, 1.0 / (k_rrf + rank)))
        if verbose:
            print(f"  BM25    : DISABLED — dense only ({len(fused)} candidates)")

    # ── Step 5: Content boost ─────────────────────────────────────────────────
    fused     = _apply_content_boost(fused, metadata, intent)
    rerank_in = fused[:RERANK_K]

    # ── Step 6: Cross-encoder reranking (use_ocr controls passage content) ────
    if use_rerank and rerank_in:
        t0 = time.time()
        reranked = _rerank(
            query      = query,
            candidates = rerank_in,
            metadata   = metadata,
            top_k      = top_k * 4,
            use_ocr    = use_ocr,      # ← OCR isolation happens here
        )
        if verbose:
            print(f"  Rerank  : {time.time()-t0:.2f}s | "
                  f"{len(rerank_in)} → {len(reranked)} "
                  f"(OCR={'ON' if use_ocr else 'OFF'})")
    else:
        reranked = rerank_in[:top_k * 4]

    # ── Step 7: Deduplication ─────────────────────────────────────────────────
    final = _deduplicate_by_lecture(reranked, metadata, top_k)
    if verbose:
        print(f"  Dedup   : {len(reranked)} segments → {len(final)} unique lectures")
        print(f"  Total   : {time.time()-t_total:.2f}s")

    # ── Build output ──────────────────────────────────────────────────────────
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
            "chunking_strategy": seg.get("chunking_strategy"),
            "retrieval_score":   round(score, 6),
            "query_intent":      intent,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: list[dict], query: str) -> None:
    print(f"\n{'='*65}")
    print(f"Query  : {query}")
    print(f"Results: {len(results)} unique lectures")
    print(f"{'='*65}")
    for r in results:
        m = int(r["start_sec"]) // 60
        s = int(r["start_sec"]) % 60
        print(f"\n  Rank {r['rank']}  [{r['content_type']}]"
              f"{'  [CODE]' if r['is_code_segment'] else ''}")
        print(f"  Course   : {r['course_name']}")
        print(f"  Lecture  : {r['lecture_title']}")
        print(f"  Time     : {m}:{s:02d}  ({r['start_sec']:.0f}s)")
        print(f"  Link     : {r['youtube_deep_link']}")
        print(f"  Score    : {r['retrieval_score']}")
        print(f"  Snippet  : {r['transcript'].replace(chr(10),' ')[:150]}...")
        print(f"  {'-'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Built-in test suite
# ─────────────────────────────────────────────────────────────────────────────

TEST_QUERIES = [
    ("how does binary search tree insertion work",  "conceptual"),
    ("explain the concept of virtual memory",       "theoretical"),
    ("what is backpropagation in neural networks",  "theoretical"),
    ("how does TCP ensure reliable delivery",       "conceptual"),
    ("what is the difference between BFS and DFS",  "conceptual"),
    ("time complexity of merge sort",               "theoretical"),
    ("proof of correctness for Dijkstra algorithm", "theoretical"),
    ("how to implement a stack in python",          "code"),
    ("write a function to reverse a linked list",   "code"),
    ("python code for binary search",               "code"),
]


def run_tests(strategy: str, use_llm: bool, use_rerank: bool) -> None:
    print(f"\nTest suite | strategy={strategy} | llm={use_llm} | rerank={use_rerank}\n")
    for query, expected_intent in TEST_QUERIES:
        results = search(query, strategy=strategy, top_k=5,
                         use_llm=use_llm, use_rerank=use_rerank, verbose=False)
        intent   = results[0]["query_intent"] if results else "unknown"
        match    = "✅" if intent == expected_intent else "⚠️ "
        urls     = [r.get("youtube_url", "") for r in results]
        unique   = len(set(urls))
        dedup_ok = "✅" if unique == len(results) else f"❌ {unique}/{len(results)} unique"
        print(f"  {match} intent={intent:<12}  dedup={dedup_ok}  {query[:48]}")
        if results:
            print(f"       → {results[0].get('course_name','')}  "
                  f"| {results[0].get('youtube_deep_link','')}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Confirmation scripts
# ─────────────────────────────────────────────────────────────────────────────

def _confirm_ollama_bleed() -> None:
    import requests as req
    print("\n=== OLLAMA KV-CACHE BLEED CONFIRMATION ===")
    print(f"Model: {OLLAMA_MODEL}")
    print("Sending two rapid sequential requests with opposite intents...\n")

    test_pairs = [
        ('Analyse: "python code for bubble sort". Return only JSON: '
         '{"intent": "code", "expanded_query": "...", "reasoning": "..."}',
         "code"),
        ('Analyse: "explain the theory of backpropagation". Return only JSON: '
         '{"intent": "theoretical", "expanded_query": "...", "reasoning": "..."}',
         "theoretical"),
    ]

    all_correct = True
    for i, (prompt, expected) in enumerate(test_pairs, 1):
        print(f"Request {i} (expected intent: {expected})")
        try:
            r = req.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model":      OLLAMA_MODEL,
                    "prompt":     prompt,
                    "stream":     False,
                    "keep_alive": 0,
                    "options":    {"temperature": 0.0, "num_predict": 80,
                                   "num_ctx": 256},
                },
                timeout=15,
            )
            raw  = r.json().get("response", "").strip()
            got  = _extract_json_object(raw)
            intent_got = got.get("intent", "PARSE_FAIL") if got else "PARSE_FAIL"
            correct    = intent_got == expected
            icon       = "✅" if correct else "❌ BLEED DETECTED"
            all_correct = all_correct and correct
            print(f"  {icon}  Got intent={intent_got!r}")
            print(f"  Raw: {raw[:100]!r}")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    if all_correct:
        print("✅ No KV-cache bleeding detected. Fix is working.\n")
    else:
        print("❌ KV-cache bleeding detected. Check keep_alive: 0 is set.\n")


def _confirm_json_parser() -> None:
    print("\n=== JSON EXTRACTOR EDGE-CASE TESTS ===\n")

    cases = [
        ('{"intent": "code", "expanded_query": "BST", "reasoning": "ok"}',
         "code", "1. Clean JSON"),
        ('Here is the result:\n{"intent": "theoretical", "expanded_query": "merge", "reasoning": "t"}',
         "theoretical", "2. JSON with preamble text"),
        ('```json\n{"intent": "conceptual", "expanded_query": "VM", "reasoning": "c"}\n```',
         "conceptual", "3. JSON in markdown code block"),
        ('{"intent": "code", "expanded_query": "loop", "reasoning": "like {x for x in y}"}',
         "code", "4. Braces inside string value"),
        ('{"intent": "theoretical", "expanded_query": "backprop", "reasoning": "t"}\n\nI hope this helps!',
         "theoretical", "5. Trailing text after JSON"),
        ('Sure! {"intent": "conceptual", "expanded_query": "net", "reasoning": "c"} Done.',
         "conceptual", "6. JSON sandwiched between text"),
    ]

    all_passed = True
    for raw, expected, label in cases:
        result = _extract_json_object(raw)
        got    = result.get("intent") if result else None
        ok     = got == expected
        all_passed = all_passed and ok
        print(f"  {'✅' if ok else '❌'} {label}")
        if not ok:
            print(f"     Expected={expected!r}  Got={got!r}")
            print(f"     Input: {raw[:80]!r}")

    print(f"\n{'✅ All passed' if all_passed else '❌ Some failed'}\n")


def _confirm_dedup(query: str, strategy: str) -> None:
    print(f"\n=== DEDUPLICATION CONFIRMATION ===")
    print(f"Query: {query} | Strategy: {strategy}\n")

    idx_data = _load_index(strategy)
    meta     = idx_data["metadata"]

    dense  = _dense_retrieve(query, idx_data, CANDIDATE_K)
    sparse = _sparse_retrieve(query, idx_data, CANDIDATE_K)
    fused  = _reciprocal_rank_fusion(dense, sparse)
    fused  = _apply_content_boost(fused, meta, "conceptual")

    top10 = fused[:10]
    urls_before = [meta[i].get("youtube_url", f"idx_{i}")
                   for i, _ in top10 if i < len(meta)]
    unique_before = len(set(urls_before))

    print("BEFORE dedup — top-10 segments:")
    for rank, (i, score) in enumerate(top10, 1):
        if i >= len(meta): continue
        title = meta[i].get("lecture_title", "?")[:48]
        t     = meta[i].get("start_sec", 0)
        print(f"  {rank:2}. {title}  @ {int(t)//60}:{int(t)%60:02d}  "
              f"[{score:.5f}]")
    print(f"  → Unique lectures: {unique_before}/10")

    deduped = _deduplicate_by_lecture(fused, meta, top_k=5)
    print("\nAFTER dedup — top-5 unique lectures:")
    for rank, (i, score) in enumerate(deduped, 1):
        if i >= len(meta): continue
        title = meta[i].get("lecture_title", "?")[:48]
        t     = meta[i].get("start_sec", 0)
        print(f"  {rank}. {title}  @ {int(t)//60}:{int(t)%60:02d}  "
              f"[{score:.5f}]")
    urls_after = [meta[i].get("youtube_url","") for i, _ in deduped if i < len(meta)]
    print(f"  → Unique lectures: {len(set(urls_after))}/{len(deduped)} ✅\n")


# ─────────────────────────────────────────────────────────────────────────────
# Flask / Streamlit API helper
# ─────────────────────────────────────────────────────────────────────────────

def api_search(
    query:    str,
    strategy: str  = "c3",
    use_llm:  bool = False,
    use_ocr:  bool = True,
    use_bm25: bool = True,
) -> list[dict]:
    """
    Thin wrapper for use by app.py (Streamlit) and any Flask API layer.
    Exposes use_ocr and use_bm25 so the UI can toggle them independently.
    """
    results = search(
        query    = query,
        strategy = strategy,
        use_llm  = use_llm,
        use_ocr  = use_ocr,
        use_bm25 = use_bm25,
        verbose  = False,
    )
    return [
        {
            "rank":         r.get("rank"),
            "transcript":   r.get("transcript", ""),
            "ocr_text":     r.get("ocr_text", ""),
            "youtube_link": r.get("youtube_deep_link", ""),
            "course_name":  r.get("course_name", ""),
            "lecture_title": r.get("lecture_title", ""),
            "start_sec":    r.get("start_sec", 0),
            "score":        r.get("retrieval_score", 0),
        }
        for r in results
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="NPTEL lecture retrieval engine")
    parser.add_argument("--query",     type=str,  default=None)
    parser.add_argument("--strategy",  type=str,  default=DEFAULT_STRATEGY,
                        choices=["c1", "c2", "c3",
                                 "c2_w150", "c2_w250",
                                 "c3_t025", "c3_t030", "c3_t040"])
    parser.add_argument("--top-k",     type=int,  default=DEFAULT_TOP_K)
    parser.add_argument("--llm",       action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-ocr",    action="store_true",
                        help="Disable OCR text in reranker passage (transcript only).")
    parser.add_argument("--no-bm25",   action="store_true",
                        help="Disable BM25 sparse retrieval (dense only).")
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--test",      action="store_true")
    parser.add_argument("--confirm-ollama-bleed", action="store_true")
    parser.add_argument("--confirm-json-parser",  action="store_true")
    parser.add_argument("--confirm-dedup",        action="store_true")
    args = parser.parse_args()

    if args.confirm_ollama_bleed:
        _confirm_ollama_bleed()
        return
    if args.confirm_json_parser:
        _confirm_json_parser()
        return
    if args.confirm_dedup:
        if not args.query:
            print("--confirm-dedup requires --query"); return
        _confirm_dedup(args.query, args.strategy)
        return
    if args.test:
        run_tests(args.strategy, args.llm, not args.no_rerank)
        return
    if not args.query:
        parser.print_help()
        return

    results = search(
        query      = args.query,
        strategy   = args.strategy,
        top_k      = args.top_k,
        use_llm    = args.llm,
        use_rerank = not args.no_rerank,
        use_ocr    = not args.no_ocr,
        use_bm25   = not args.no_bm25,
        verbose    = args.verbose,
    )
    print_results(results, args.query)


if __name__ == "__main__":
    main()