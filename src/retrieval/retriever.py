"""
retriever.py  —  Phase 2B: Query Engine
========================================
Implements the full retrieval pipeline:

  1. [Optional] LLM query analysis via Ollama
       - Detects query intent (code / theoretical / conceptual)
       - Expands query with synonyms for better recall
  2. BGE-large query embedding  (dense)
  3. BM25 keyword scoring       (sparse)
  4. Reciprocal Rank Fusion     (merge dense + sparse)
  5. Content-type score boost   (code/theory differentiation)
  6. Cross-encoder re-ranking   (top-50 → top-5 with ms-marco-MiniLM)

Can be used as:
  a) Python module  — import and call search()
  b) CLI tool       — python retriever.py --query "..." --strategy c2
  c) Test harness   — python retriever.py --test

LLM integration (Ollama):
  Set USE_LLM=true in .env and make sure Ollama is running.
  The LLM adds ~0.5-1s latency per query but improves precision on
  ambiguous queries. It is off by default so the system works without it.

Usage
-----
    # Basic search
    python retriever.py --query "how does binary search tree insertion work"

    # Specify strategy
    python retriever.py --query "BST insertion" --strategy c3

    # With LLM analysis enabled (Ollama must be running)
    python retriever.py --query "BST insertion" --llm

    # Return more results
    python retriever.py --query "BST insertion" --top-k 10

    # Run built-in test suite
    python retriever.py --test

    # Disable cross-encoder reranking (faster, slightly less precise)
    python retriever.py --query "BST insertion" --no-rerank
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any

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
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL",    "BAAI/bge-large-en-v1.5")
RERANKER_MODEL     = os.getenv("RERANKER_MODEL",     "cross-encoder/ms-marco-MiniLM-L-6-v2")
EMBEDDING_DEVICE   = os.getenv("EMBEDDING_DEVICE",   "cuda")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL",       "llama3.2:3b")
OLLAMA_HOST        = os.getenv("OLLAMA_HOST",        "http://localhost:11434")

# ── retrieval config ──────────────────────────────────────────────────────────
DEFAULT_STRATEGY   = os.getenv("DEFAULT_STRATEGY",   "c2")
CANDIDATE_K        = 100     # dense + sparse each retrieve this many
RERANK_K           = 50      # top-N after fusion that go to cross-encoder
DEFAULT_TOP_K      = 5       # final results returned to user

# ── content-type boost factors ────────────────────────────────────────────────
# When query intent matches segment content_type, multiply its RRF score.
# This is the mechanism that differentiates code vs theory queries.
# Values are intentionally modest — boost, not override.
BOOST_CODE         = 1.4     # code query → boost is_code_segment results
BOOST_THEORY       = 1.25    # theory query → boost theoretical results
BOOST_CONCEPTUAL   = 1.1     # conceptual query → slight boost conceptual


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (cached — loaded once per process)
# ─────────────────────────────────────────────────────────────────────────────

_embed_model    = None
_rerank_model   = None

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
    Returns {"faiss": index, "metadata": list, "bm25": BM25, "corpus": list}
    Loads from disk on first call, then caches in memory.
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
    if not bm25_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {bm25_path}\n"
            f"Run: python bm25_builder.py --strategy {strategy}"
        )

    print(f"  Loading indexes for strategy={strategy} ...", flush=True)
    t0 = time.time()

    faiss_index = faiss.read_index(str(faiss_path))
    metadata    = json.loads(meta_path.read_text(encoding="utf-8"))

    with open(bm25_path, "rb") as fh:
        bm25_data = pickle.load(fh)

    elapsed = time.time() - t0
    print(f"  Loaded {faiss_index.ntotal:,} vectors in {elapsed:.1f}s", flush=True)

    result = {
        "faiss":    faiss_index,
        "metadata": metadata,
        "bm25":     bm25_data["bm25"],
        "corpus":   bm25_data["corpus"],
    }
    _index_cache[strategy] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BM25 tokeniser  (must match bm25_builder.py tokenise())
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
# Step 1: LLM query analysis (optional)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_query_with_llm(query: str) -> dict:
    """
    Uses local Ollama LLM to:
      - Classify query intent: "code", "theoretical", or "conceptual"
      - Generate expanded query with synonyms for better recall

    Returns:
        {
            "intent":         "code" | "theoretical" | "conceptual",
            "expanded_query": "original query + synonyms",
            "reasoning":      "brief explanation"
        }

    Falls back gracefully if Ollama is not running or times out.

    WHY THIS HELPS:
      "how to write a for loop" → classified as "code"
        → retriever boosts is_code_segment results
        → user gets code slides, not theory slides
      "explain backpropagation" → classified as "theoretical"
        → retriever boosts theoretical content_type results

    The LLM also expands the query:
      "BST insertion" → "BST insertion binary search tree insert node key"
    This improves BM25 recall significantly for short queries.
    """

    prompt = f"""You are a query analyser for an educational video retrieval system.
    
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
    - "conceptual": user wants to understand how something works, examples, intuition"""
    
    print(f"🚀 Using model: {OLLAMA_MODEL}")

    try:
        import requests
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 150},
            },
            timeout=10,
        )
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()

        # Extract JSON from response (LLM may add extra text despite instructions)
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # Validate and sanitise
            intent = result.get("intent", "conceptual")
            if intent not in ("code", "theoretical", "conceptual"):
                intent = "conceptual"
            return {
                "intent":         intent,
                "expanded_query": result.get("expanded_query", query),
                "reasoning":      result.get("reasoning", ""),
            }

    except Exception as e:
        print("❌ LLM ERROR:", e)
        # Silent fallback — never crash the retrieval pipeline due to LLM issues
        pass

    print("==============================Fallback strategy used ==================================")
    # Fallback: no LLM analysis
    return {
        "intent":         "conceptual",
        "expanded_query": query,
        "reasoning":      "LLM unavailable — using default intent",
    }

def _detect_intent_heuristic(query: str) -> str:
    """
    Fast rule-based intent detection (used when LLM is disabled).
    Checks for code-signal keywords in the query.
    """
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
# Step 2: Dense retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _dense_retrieve(query: str, index_data: dict, k: int) -> list[tuple[int, float]]:
    """
    Embeds query with BGE-large and searches FAISS.
    Returns list of (row_index, score) pairs, sorted by score desc.
    BGE-large requires "query: " prefix on query strings.
    """
    import numpy as np
    model = _get_embed_model()
    q_vec = model.encode(
        ["query: " + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, indices = index_data["faiss"].search(q_vec, k)
    # scores[0] and indices[0] are the results for our single query
    return list(zip(indices[0].tolist(), scores[0].tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Sparse (BM25) retrieval
# ─────────────────────────────────────────────────────────────────────────────

def _sparse_retrieve(query: str, index_data: dict, k: int) -> list[tuple[int, float]]:
    """
    Tokenises query and scores all documents with BM25.
    Returns top-k (row_index, score) pairs.
    """
    import numpy as np
    tokens = _tokenise(query)
    if not tokens:
        return []

    scores  = index_data["bm25"].get_scores(tokens)
    top_k   = min(k, len(scores))
    indices = scores.argsort()[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in indices if scores[i] > 0]


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    dense_results:  list[tuple[int, float]],
    sparse_results: list[tuple[int, float]],
    k_rrf: int = 60,
) -> list[tuple[int, float]]:
    """
    Combines dense and sparse rankings using Reciprocal Rank Fusion.

    RRF score for document d = Σ  1 / (k + rank(d))
    where the sum is over each ranking list that contains d.
    k=60 is the standard constant from the original RRF paper (Cormack 2009).

    This is parameter-free — no manual weight tuning needed.
    Dense and sparse are weighted equally by default.
    """
    rrf_scores: dict[int, float] = {}

    for rank, (idx, _score) in enumerate(dense_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank)

    for rank, (idx, _score) in enumerate(sparse_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank)

    # Sort by fused score, highest first
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Content-type boost
# ─────────────────────────────────────────────────────────────────────────────

def _apply_content_boost(
    fused: list[tuple[int, float]],
    metadata: list[dict],
    intent: str,
) -> list[tuple[int, float]]:
    """
    Multiplies RRF scores for segments whose content_type matches query intent.

    This is the key mechanism for code vs theory differentiation:
      - A "code" query boosts segments with is_code_segment=True
      - A "theoretical" query boosts segments with content_type="theoretical"

    The boost is additive to the RRF score so it cannot completely override
    retrieval ranking — it nudges, not overrides. Good matching segments
    that are already in the top-50 get promoted; irrelevant segments cannot
    be boosted into the top results.
    """
    boosted = []
    for idx, score in fused:
        if idx >= len(metadata):
            boosted.append((idx, score))
            continue

        seg   = metadata[idx]
        ctype = seg.get("content_type", "conceptual")
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
# Step 6: Cross-encoder re-ranking
# ─────────────────────────────────────────────────────────────────────────────

def _rerank(
    query:    str,
    candidates: list[tuple[int, float]],
    metadata:   list[dict],
    top_k:    int,
) -> list[tuple[int, float]]:
    """
    Re-ranks the top-N candidates using a cross-encoder.

    The cross-encoder (ms-marco-MiniLM-L-6-v2) sees the full
    (query, passage) pair simultaneously — much more accurate than
    bi-encoder cosine similarity but too slow to run on thousands of
    candidates. Running it on the top-50 fused candidates is the
    standard two-stage retrieval approach.

    The passage fed to the cross-encoder combines OCR + transcript
    (same as what gets embedded, without the structured prefix).
    """
    reranker = _get_rerank_model()

    pairs = []
    valid_indices = []

    for idx, _score in candidates:
        if idx >= len(metadata):
            continue
        seg = metadata[idx]
        # Build passage for cross-encoder
        ocr        = seg.get("ocr_text", "").replace("\n---\n", " ").replace("\n", " ").strip()
        transcript = seg.get("transcript", "").strip()
        if ocr and not seg.get("ocr_failed", False):
            passage = f"{ocr} {transcript}"
        else:
            passage = transcript
        pairs.append([query, passage[:512]])   # cross-encoder has token limit
        valid_indices.append(idx)

    if not pairs:
        return candidates[:top_k]

    scores      = reranker.predict(pairs)
    reranked    = sorted(zip(valid_indices, scores.tolist()),
                         key=lambda x: x[1], reverse=True)
    return reranked[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Main search function (public API)
# ─────────────────────────────────────────────────────────────────────────────

def search(
    query:      str,
    strategy:   str  = DEFAULT_STRATEGY,
    top_k:      int  = DEFAULT_TOP_K,
    use_llm:    bool = False,
    use_rerank: bool = True,
    verbose:    bool = False,
) -> list[dict]:
    """
    Full retrieval pipeline. Returns top_k result dicts.

    Each result dict contains:
        segment_id, course_name, lecture_title, youtube_deep_link,
        start_sec, transcript, ocr_text, content_type,
        is_code_segment, scores (dict with pipeline stage scores)

    Args:
        query       — natural language query string
        strategy    — "c1", "c2", or "c3" (chunking strategy)
        top_k       — number of results to return
        use_llm     — whether to use Ollama LLM for query analysis
        use_rerank  — whether to use cross-encoder reranking
        verbose     — print timing breakdown

    Pipeline timing estimates on RTX 3060:
        LLM analysis  : ~0.5-1.0s  (when enabled)
        Dense embed   : ~0.05s
        Dense search  : ~0.01s
        BM25 search   : ~0.1s
        RRF + boost   : ~0.01s
        Cross-encoder : ~0.5-1.0s  (50 pairs)
        Total         : ~0.7-2.0s
    """
    t_total = time.time()

    # ── Step 1: Query analysis ─────────────────────────────────────────────
    if use_llm:
        t0 = time.time()
        analysis = analyse_query_with_llm(query)
        print("--LLM returned this query --",analysis)
        intent         = analysis["intent"]
        expanded_query = analysis["expanded_query"]
        if verbose:
            print(f"  LLM analysis : {time.time()-t0:.2f}s | "
                  f"intent={intent} | expanded='{expanded_query[:60]}'")
    else:
        intent         = _detect_intent_heuristic(query)
        expanded_query = query
        if verbose:
            print(f"  Heuristic intent: {intent}")

    # ── Load indexes ───────────────────────────────────────────────────────
    index_data = _load_index(strategy)
    metadata   = index_data["metadata"]

    # ── Step 2: Dense retrieval ────────────────────────────────────────────
    t0 = time.time()
    dense_results = _dense_retrieve(expanded_query, index_data, CANDIDATE_K)
    if verbose:
        print(f"  Dense retrieval: {time.time()-t0:.2f}s | "
              f"{len(dense_results)} candidates")

    # ── Step 3: Sparse (BM25) retrieval ───────────────────────────────────
    t0 = time.time()
    sparse_results = _sparse_retrieve(expanded_query, index_data, CANDIDATE_K)
    if verbose:
        print(f"  Sparse BM25    : {time.time()-t0:.2f}s | "
              f"{len(sparse_results)} candidates")

    # ── Step 4: Reciprocal Rank Fusion ─────────────────────────────────────
    t0 = time.time()
    fused = _reciprocal_rank_fusion(dense_results, sparse_results)
    if verbose:
        print(f"  RRF fusion     : {time.time()-t0:.3f}s | "
              f"{len(fused)} unique candidates")

    # ── Step 5: Content-type boost ─────────────────────────────────────────
    fused = _apply_content_boost(fused, metadata, intent)
    fused_top = fused[:RERANK_K]

    # ── Step 6: Cross-encoder re-rank ──────────────────────────────────────
    if use_rerank and fused_top:
        t0 = time.time()
        final_ranked = _rerank(query, fused_top, metadata, top_k)
        if verbose:
            print(f"  Cross-encoder  : {time.time()-t0:.2f}s | "
                  f"top-{len(fused_top)} → top-{len(final_ranked)}")
    else:
        final_ranked = fused_top[:top_k]

    if verbose:
        print(f"  Total latency  : {time.time()-t_total:.2f}s")

    # ── Build output ───────────────────────────────────────────────────────
    results = []
    for rank, (idx, score) in enumerate(final_ranked, start=1):
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
            "transcript":        seg.get("transcript", "")[:400],  # truncate for display
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
    print(f"Query: {query}")
    print(f"{'='*65}")
    for r in results:
        minutes = int(r["start_sec"]) // 60
        seconds = int(r["start_sec"]) % 60
        print(f"\n  Rank {r['rank']}  [{r['content_type']}]"
              f"{'  [CODE]' if r['is_code_segment'] else ''}")
        print(f"  Course   : {r['course_name']}")
        print(f"  Lecture  : {r['lecture_title']}")
        print(f"  Time     : {minutes}:{seconds:02d}  ({r['start_sec']:.0f}s)")
        print(f"  Link     : {r['youtube_deep_link']}")
        print(f"  Score    : {r['retrieval_score']}")
        transcript = r["transcript"].replace("\n", " ")
        print(f"  Snippet  : {transcript[:150]}...")
        print(f"  {'-'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Built-in test suite
# ─────────────────────────────────────────────────────────────────────────────

TEST_QUERIES = [
    # Conceptual
    ("how does binary search tree insertion work",      "conceptual"),
    ("explain the concept of virtual memory",           "theoretical"),
    ("what is backpropagation in neural networks",      "theoretical"),
    ("how does TCP ensure reliable delivery",           "conceptual"),
    ("what is the difference between BFS and DFS",     "conceptual"),
    # Theoretical
    ("time complexity of merge sort",                  "theoretical"),
    ("proof of correctness for Dijkstra algorithm",    "theoretical"),
    # Code
    ("how to implement a stack in python",             "code"),
    ("write a function to reverse a linked list",      "code"),
    ("python code for binary search",                  "code"),
]

def run_tests(strategy: str, use_llm: bool, use_rerank: bool) -> None:
    print(f"\nRunning test suite on strategy={strategy} | "
          f"llm={use_llm} | rerank={use_rerank}\n")

    for query, expected_intent in TEST_QUERIES:
        results = search(
            query,
            strategy=strategy,
            top_k=3,
            use_llm=use_llm,
            use_rerank=use_rerank,
            verbose=False,
        )
        actual_intent = results[0]["query_intent"] if results else "unknown"
        intent_match  = "✅" if actual_intent == expected_intent else "⚠️ "
        top_course    = results[0]["course_name"] if results else "no results"
        top_link      = results[0]["youtube_deep_link"] if results else ""
        print(f"  {intent_match} [{actual_intent:<12}] {query[:50]}")
        print(f"       → {top_course}  |  {top_link}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lecture video retrieval engine"
    )
    parser.add_argument("--query",     type=str, default=None,
                        help="Natural language query string.")
    parser.add_argument("--strategy",  type=str, default=DEFAULT_STRATEGY,
                        choices=["c1", "c2", "c3"],
                        help=f"Chunking strategy to search (default {DEFAULT_STRATEGY}).")
    parser.add_argument("--top-k",    type=int, default=DEFAULT_TOP_K,
                        help=f"Number of results to return (default {DEFAULT_TOP_K}).")
    parser.add_argument("--llm",       action="store_true",
                        help="Enable LLM query analysis via Ollama.")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Disable cross-encoder re-ranking (faster).")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print timing breakdown.")
    parser.add_argument("--test",      action="store_true",
                        help="Run built-in test query suite.")
    args = parser.parse_args()

    if args.test:
        run_tests(
            strategy   = args.strategy,
            use_llm    = args.llm,
            use_rerank = not args.no_rerank,
        )
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
        verbose    = args.verbose,
    )
    print_results(results, args.query)


# for Flask application 
def api_search(query, strategy="c3", use_llm=False):
    results = search(
        query=query,
        strategy=strategy,
        use_llm=use_llm,
        verbose=False
    )

    formatted = []
    for r in results:
        formatted.append({
            "transcript": r.get("transcript", ""),
            "ocr_text": r.get("ocr_text", ""),
            "youtube_link": r.get("youtube_deep_link", "")
        })

    return formatted

if __name__ == "__main__":
    main()
