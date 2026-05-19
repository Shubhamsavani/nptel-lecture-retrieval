"""
evaluator.py  —  Phase 3: Evaluation + Ablation Study
======================================================
Runs the full evaluation framework across two experiment groups:

GROUP 1 — System Experiments (E1–E9)
--------------------------------------
Tests chunking strategy × OCR × BM25 contributions in isolation.

  E1: C1, no OCR, no BM25   — fixed-30s, transcript only
  E2: C1 + OCR, no BM25     — fixed-30s, multimodal
  E3: C1 + OCR + BM25       — fixed-30s, full hybrid
  E4: C2, no OCR, no BM25   — utterance, transcript only
  E5: C2 + OCR, no BM25     — utterance, multimodal
  E6: C2 + OCR + BM25       — utterance, full hybrid
  E7: C3, no OCR, no BM25   — slide-boundary, transcript only
  E8: C3 + OCR, no BM25     — slide-boundary, multimodal
  E9: C3 + OCR + BM25       — slide-boundary, full hybrid  ← Full system

GROUP 2 — Chunk Parameter Sensitivity (NEW)
--------------------------------------------
Tests different chunk sizes/thresholds with OCR and BM25 BOTH DISABLED
to isolate the pure chunking effect.

  C2-150  : C2 utterance, 150-word target
  C2-200  : C2 utterance, 200-word target  (default)
  C2-250  : C2 utterance, 250-word target
  C3-0.25 : C3 slide-boundary, threshold=0.25
  C3-0.30 : C3 slide-boundary, threshold=0.30
  C3-0.40 : C3 slide-boundary, threshold=0.40

WHY OCR AND BM25 ARE DISABLED FOR GROUP 2:
  The goal is to measure the effect of chunking parameters only.
  Including OCR or BM25 would confound the results — any difference
  between C2-150 and C2-200 could be due to BM25 score distributions
  rather than chunk granularity. Disabling both isolates the variable.

EVALUATION SET
--------------
Loaded from data/eval/annotations.jsonl (status="selected" entries only).

METRICS
-------
  MRR        : Mean Reciprocal Rank  (primary metric)
  Recall@5   : Fraction of queries where correct answer is in top-5
  Recall@10  : Fraction of queries where correct answer is in top-10
  LLM judge  : Llama-via-Ollama scores top-1 result (0–3). Averaged.

Usage
-----
    # Full evaluation — both groups
    python evaluator.py

    # Single system experiment
    python evaluator.py --experiment E7

    # Group 1 only (E1-E9)
    python evaluator.py --group1-only

    # Group 2 only (chunk sensitivity)
    python evaluator.py --group2-only

    # Ablation dataset statistics only (no retrieval)
    python evaluator.py --ablation-only

    # Show queries missing timestamps
    python evaluator.py --show-unfilled

    # Skip LLM judge (faster)
    python evaluator.py --no-llm-judge

Output
------
    data/eval/
        results_E1.json ... results_E9.json
        sensitivity_C2-150.json ... sensitivity_C3-0.40.json
        ablation_results.json
        eval_summary.csv        ← paste into Excel for graphs
        llm_judge_scores.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import csv
from pathlib import Path
from datetime import datetime

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
PROCESSED    = PROJECT_ROOT / "data" / "processed"
EVAL_DIR     = PROJECT_ROOT / "data" / "eval"

sys.path.insert(0, str(Path(__file__).resolve().parent))

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 1 — System Experiment Definitions (E1–E9)
# ─────────────────────────────────────────────────────────────────────────────
#
# Design matrix:
#   Rows    = chunking strategy (C1 / C2 / C3)
#   Columns = pipeline config   (no OCR no BM25 / OCR only / OCR + BM25)
#
# Each cell must produce genuinely different retrieval behaviour:
#   use_ocr=False  → reranker sees transcript-only passages
#   use_ocr=True   → reranker sees OCR+transcript passages
#   use_bm25=False → only FAISS dense results used
#   use_bm25=True  → FAISS + BM25 fused via RRF
#
# Strategy key maps to FAISS/BM25 index files:
#   c1  → faiss_c1.index  / bm25_c1.pkl
#   c2  → faiss_c2.index  / bm25_c2.pkl
#   c3  → faiss_c3.index  / bm25_c3.pkl
#
EXPERIMENTS = {
    # ── C1: fixed-30s chunks ──────────────────────────────────────────────────
    "E1": {
        "strategy": "c1",
        "use_ocr":  False,
        "use_bm25": False,
        "label":    "C1 fixed-30s | transcript only",
        "group":    1,
    },
    "E2": {
        "strategy": "c1",
        "use_ocr":  True,
        "use_bm25": False,
        "label":    "C1 fixed-30s | transcript + OCR",
        "group":    1,
    },
    "E3": {
        "strategy": "c1",
        "use_ocr":  True,
        "use_bm25": True,
        "label":    "C1 fixed-30s | transcript + OCR + BM25",
        "group":    1,
    },

    # ── C2: utterance chunks ──────────────────────────────────────────────────
    "E4": {
        "strategy": "c2",
        "use_ocr":  False,
        "use_bm25": False,
        "label":    "C2 utterance | transcript only",
        "group":    1,
    },
    "E5": {
        "strategy": "c2",
        "use_ocr":  True,
        "use_bm25": False,
        "label":    "C2 utterance | transcript + OCR",
        "group":    1,
    },
    "E6": {
        "strategy": "c2",
        "use_ocr":  True,
        "use_bm25": True,
        "label":    "C2 utterance | transcript + OCR + BM25",
        "group":    1,
    },

    # ── C3: slide-boundary chunks ─────────────────────────────────────────────
    "E7": {
        "strategy": "c3",
        "use_ocr":  False,
        "use_bm25": False,
        "label":    "C3 slide-boundary | transcript only",
        "group":    1,
    },
    "E8": {
        "strategy": "c3",
        "use_ocr":  True,
        "use_bm25": False,
        "label":    "C3 slide-boundary | transcript + OCR",
        "group":    1,
    },
    "E9": {
        "strategy": "c3",
        "use_ocr":  True,
        "use_bm25": True,
        "label":    "C3 slide-boundary | transcript + OCR + BM25  ← Full system",
        "group":    1,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 2 — Chunk Parameter Sensitivity Definitions
# ─────────────────────────────────────────────────────────────────────────────
#
# OCR and BM25 are ALWAYS disabled here.
# Reason: isolate the effect of chunking parameters only.
#
# Strategy key maps to variant FAISS/BM25 indexes:
#   c2_w150  → faiss_c2_w150.index  / bm25_c2_w150.pkl
#   c2_w200  → faiss_c2.index       / bm25_c2.pkl        (reuses default c2)
#   c2_w250  → faiss_c2_w250.index  / bm25_c2_w250.pkl
#   c3_t025  → faiss_c3_t025.index  / bm25_c3_t025.pkl
#   c3_t030  → faiss_c3_t030.index  / bm25_c3_t030.pkl
#   c3_t040  → faiss_c3_t040.index  / bm25_c3_t040.pkl
#
# Note: c2_w200 deliberately reuses the default c2 index so that the
# 200-word result is directly comparable to E4 (same index, same config).
#
SENSITIVITY_VARIANTS = {
    "C2-150": {
        "strategy":   "c2_w150",
        "use_ocr":    False,   # MUST be False — isolate chunking only
        "use_bm25":   False,   # MUST be False — isolate chunking only
        "chunk_type": "C2",
        "parameter":  "150w",
        "label":      "C2 utterance | 150-word target | no OCR | no BM25",
        "group":      2,
    },
    "C2-200": {
        "strategy":   "c2",    # reuses default C2 index
        "use_ocr":    False,
        "use_bm25":   False,
        "chunk_type": "C2",
        "parameter":  "200w",
        "label":      "C2 utterance | 200-word target | no OCR | no BM25",
        "group":      2,
    },
    "C2-250": {
        "strategy":   "c2_w250",
        "use_ocr":    False,
        "use_bm25":   False,
        "chunk_type": "C2",
        "parameter":  "250w",
        "label":      "C2 utterance | 250-word target | no OCR | no BM25",
        "group":      2,
    },
    "C3-0.25": {
        "strategy":   "c3_t025",
        "use_ocr":    False,
        "use_bm25":   False,
        "chunk_type": "C3",
        "parameter":  "t=0.25",
        "label":      "C3 slide-boundary | threshold=0.25 | no OCR | no BM25",
        "group":      2,
    },
    "C3-0.30": {
        "strategy":   "c3",
        "use_ocr":    False,
        "use_bm25":   False,
        "chunk_type": "C3",
        "parameter":  "t=0.30",
        "label":      "C3 slide-boundary | threshold=0.30 | no OCR | no BM25",
        "group":      2,
    },
    "C3-0.40": {
        "strategy":   "c3_t040",
        "use_ocr":    False,
        "use_bm25":   False,
        "chunk_type": "C3",
        "parameter":  "t=0.40",
        "label":      "C3 slide-boundary | threshold=0.40 | no OCR | no BM25",
        "group":      2,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Ablation file variants (dataset statistics only — no retrieval)
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_VARIANTS = [
    {"file": "segments_c1.jsonl",      "label": "C1 (30s)"},
    {"file": "segments_c2.jsonl",      "label": "C2 (200w)"},
    {"file": "segments_c2_w150.jsonl", "label": "C2 (150w)"},
    {"file": "segments_c2_w250.jsonl", "label": "C2 (250w)"},
    {"file": "segments_c3_t025.jsonl", "label": "C3 (t=0.25)"},
    {"file": "segments_c3_t030.jsonl", "label": "C3 (t=0.30)"},
    {"file": "segments_c3_t040.jsonl", "label": "C3 (t=0.40)"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Annotation loader
# ─────────────────────────────────────────────────────────────────────────────

def load_annotations(path: str) -> list[dict]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("status") != "selected":
                continue
            queries.append({
                "id":                 item["id"],
                "query":              item["query"],
                "type":               item.get("type", "conceptual"),
                "expected_course":    item["expected_course"],
                "expected_lecture":   item.get("expected_lecture", 0),
                "expected_start_sec": item.get("expected_start_sec", 0),
                "notes":              item.get("notes", ""),
            })
    return queries


EVAL_QUERIES = load_annotations(str(EVAL_DIR / "annotations.jsonl"))
print(f"Loaded {len(EVAL_QUERIES)} queries from annotation file")


# ─────────────────────────────────────────────────────────────────────────────
# LLM-as-judge
# ─────────────────────────────────────────────────────────────────────────────

def llm_judge_score(query: str, transcript: str, course: str, lecture: str) -> int:
    """
    Uses Llama 3.2:3b via Ollama to score the top-1 retrieved result.

    Scoring rubric:
        3 = Perfect match — directly answers the query
        2 = Partial match — related content, partially answers
        1 = Related       — same topic but does not answer query
        0 = Irrelevant    — unrelated to query

    Returns -1 on failure (excluded from average).
    """
    prompt = f"""You are evaluating a lecture video retrieval system.

Query: "{query}"

Retrieved transcript (from {course} — {lecture}):
"{transcript[:400]}"

Score how well this transcript answers the query:
  3 = Perfect: directly and completely answers the query
  2 = Partial: relevant content but only partially answers
  1 = Related: same general topic but does not answer the query
  0 = Irrelevant: unrelated to the query

Respond with ONLY a single digit (0, 1, 2, or 3). Nothing else."""

    try:
        import requests
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 5},
            },
            timeout=15,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()
        match = next((c for c in raw if c in "0123"), None)
        return int(match) if match else -1
    except Exception:
        return -1


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

MATCH_TOLERANCE_SEC = 90  # within 90 seconds = correct answer

# Maps course IDs to substrings expected in course_name field
COURSE_ID_MAP = {
    "dsa":  "algorithms and analysis",
    "daa":  "design and analysis",
    "dl":   "deep learning",
    "os":   "operating systems",
    "dbms": "database management",
    "cv":   "computer vision",
    "coa":  "computer architecture",
    "ml":   "machine learning",
    "cn":   "computer networks",
}


def _is_correct(result: dict, query_meta: dict) -> bool:
    """
    Returns True if a retrieved result matches the expected answer.

    Matching criteria (both must be satisfied):
      1. course_name contains the expected course substring
      2. start_sec is within MATCH_TOLERANCE_SEC of expected_start_sec
         (only checked when expected_start_sec > 0)

    If expected_start_sec == 0 (not yet annotated), only course match is used.
    """
    if result.get("course_name") is None:
        return False

    expected_course = query_meta["expected_course"]
    course_name     = result.get("course_name", "").lower()
    expected_substr = COURSE_ID_MAP.get(expected_course, expected_course)

    if expected_substr not in course_name:
        return False

    expected_sec = query_meta.get("expected_start_sec", 0)
    if expected_sec > 0:
        retrieved_sec = result.get("start_sec", 0)
        if abs(retrieved_sec - expected_sec) > MATCH_TOLERANCE_SEC:
            return False

    return True


def compute_metrics(
    all_results:  list[list[dict]],
    eval_queries: list[dict],
    judge_scores: list[int],
) -> dict:
    """
    Computes MRR, Recall@5, Recall@10, LLM judge average, and per-type MRR.
    """
    n       = len(eval_queries)
    rr_sum  = 0.0
    recall5 = 0
    recall10 = 0
    by_type  = {"conceptual": [], "procedural": [], "code": []}
    annotated = 0

    for i, (results, q) in enumerate(zip(all_results, eval_queries)):
        correct_rank = None
        for rank, r in enumerate(results[:10], start=1):
            if _is_correct(r, q):
                correct_rank = rank
                break

        if q.get("expected_start_sec", 0) > 0:
            annotated += 1

        qtype = q.get("type", "conceptual")
        if qtype not in by_type:
            qtype = "conceptual"

        if correct_rank is not None:
            rr_sum   += 1.0 / correct_rank
            if correct_rank <= 5:
                recall5 += 1
            if correct_rank <= 10:
                recall10 += 1
            by_type[qtype].append(1.0 / correct_rank)
        else:
            by_type[qtype].append(0.0)

    # Use annotated count as denominator if any timestamps are filled
    denom = annotated if annotated > 0 else n

    valid_judge = [s for s in judge_scores if s >= 0]
    judge_avg   = sum(valid_judge) / len(valid_judge) if valid_judge else -1

    return {
        "MRR":             round(rr_sum   / denom, 4),
        "Recall@5":        round(recall5  / denom, 4),
        "Recall@10":       round(recall10 / denom, 4),
        "LLM_judge":       round(judge_avg, 4) if judge_avg >= 0 else "N/A",
        "annotated_n":     annotated,
        "total_n":         n,
        "MRR_conceptual":  round(
            sum(by_type["conceptual"]) / max(len(by_type["conceptual"]), 1), 4),
        "MRR_procedural":  round(
            sum(by_type["procedural"]) / max(len(by_type["procedural"]), 1), 4),
        "MRR_code":        round(
            sum(by_type["code"]) / max(len(by_type["code"]), 1), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment runner (shared by both groups)
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    exp_id:        str,
    exp_config:    dict,
    eval_queries:  list[dict],
    use_llm_judge: bool = True,
    top_k:         int  = 10,
) -> dict:
    """
    Runs one experiment over all evaluation queries.

    Passes use_ocr and use_bm25 from exp_config directly into retriever.search().
    This is the critical fix: the retriever now actually respects these flags.

    Parameters
    ----------
    exp_id      : Experiment identifier string (e.g. 'E7', 'C2-150').
    exp_config  : Dict with keys: strategy, use_ocr, use_bm25, label.
    eval_queries: List of query dicts with expected answers.
    use_llm_judge: Whether to call Ollama LLM for top-1 scoring.
    top_k       : Number of results to retrieve per query.

    Returns
    -------
    metrics dict including MRR, Recall@5, Recall@10, per-type MRR, timing.
    """
    import retriever as ret

    strategy = exp_config["strategy"]
    use_ocr  = exp_config["use_ocr"]
    use_bm25 = exp_config["use_bm25"]
    label    = exp_config["label"]

    print(f"\n  Running {exp_id}: {label}")
    print(f"  strategy={strategy} | use_ocr={use_ocr} | use_bm25={use_bm25}",
          flush=True)

    all_results  = []
    judge_scores = []
    t0           = time.time()

    for i, q in enumerate(eval_queries):
        # ── Critical fix: pass use_ocr and use_bm25 to retriever ─────────────
        # Previously these flags were defined in EXPERIMENTS but never forwarded
        # to ret.search(), causing E3/E4/E5 to all run the same full pipeline.
        # Now each experiment genuinely differs:
        #   use_ocr=False  → reranker uses transcript-only passages
        #   use_bm25=False → RRF skipped, dense-only candidate list
        results = ret.search(
            query      = q["query"],
            strategy   = strategy,
            top_k      = top_k,
            use_llm    = False,     # always off during eval (consistency)
            use_rerank = True,      # always on — never disable reranking
            use_ocr    = use_ocr,   # ← was missing before
            use_bm25   = use_bm25,  # ← was missing before
            verbose    = False,
        )
        all_results.append(results)

        # LLM judge on top-1
        if use_llm_judge and results:
            top = results[0]
            score = llm_judge_score(
                query      = q["query"],
                transcript = top.get("transcript", ""),
                course     = top.get("course_name", ""),
                lecture    = top.get("lecture_title", ""),
            )
            judge_scores.append(score)
        else:
            judge_scores.append(-1)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(eval_queries)} queries done  ({elapsed:.0f}s elapsed)",
                  flush=True)

    metrics = compute_metrics(all_results, eval_queries, judge_scores)
    metrics["experiment"]   = exp_id
    metrics["label"]        = label
    metrics["strategy"]     = strategy
    metrics["use_ocr"]      = use_ocr
    metrics["use_bm25"]     = use_bm25
    metrics["elapsed_sec"]  = round(time.time() - t0, 1)
    metrics["judge_scores"] = judge_scores

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Ablation runner (dataset statistics — no retrieval needed)
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(
    eval_queries:  list[dict],
    use_llm_judge: bool = False,
) -> list[dict]:
    """
    Analyses all segment .jsonl variants and returns dataset statistics.
    No retrieval is performed — this measures coverage, chunk size, OCR rate.
    """
    results = []

    for variant in ABLATION_VARIANTS:
        fname = variant["file"]
        fpath = PROCESSED / fname
        label = variant["label"]

        if not fpath.exists():
            print(f"  [SKIP] {fname} not found", flush=True)
            results.append({"label": label, "file": fname, "status": "missing"})
            continue

        print(f"  Analysing {fname} ...", flush=True)

        segments = []
        with open(fpath, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        segments.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        n = len(segments)
        if n == 0:
            results.append({"label": label, "file": fname, "n": 0})
            continue

        avg_dur   = sum(s.get("duration_sec", 0) for s in segments) / n
        avg_words = sum(s.get("word_count",   0) for s in segments) / n
        ocr_fail  = sum(1 for s in segments if s.get("ocr_failed", False)) / n
        code_frac = sum(1 for s in segments if s.get("is_code_segment", False)) / n

        by_course: dict[str, dict] = {}
        for s in segments:
            cid = s.get("course_id", "unknown")
            if cid not in by_course:
                by_course[cid] = {"n": 0, "ocr_fail": 0, "code": 0,
                                  "dur_sum": 0.0, "words_sum": 0}
            by_course[cid]["n"]         += 1
            by_course[cid]["ocr_fail"]  += int(s.get("ocr_failed", False))
            by_course[cid]["code"]      += int(s.get("is_code_segment", False))
            by_course[cid]["dur_sum"]   += s.get("duration_sec", 0)
            by_course[cid]["words_sum"] += s.get("word_count", 0)

        course_stats = {
            cid: {
                "n":            v["n"],
                "ocr_fail_pct": round(v["ocr_fail"] / v["n"] * 100, 1),
                "code_pct":     round(v["code"]     / v["n"] * 100, 1),
                "avg_dur":      round(v["dur_sum"]  / v["n"], 1),
                "avg_words":    round(v["words_sum"] / v["n"], 1),
            }
            for cid, v in by_course.items()
        }

        results.append({
            "label":        label,
            "file":         fname,
            "n_segments":   n,
            "avg_dur_sec":  round(avg_dur, 1),
            "avg_words":    round(avg_words, 1),
            "ocr_fail_pct": round(ocr_fail * 100, 1),
            "code_pct":     round(code_frac * 100, 1),
            "by_course":    course_stats,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_group1_table(all_metrics: list[dict]) -> None:
    """Prints the 3×3 design matrix for Group 1 experiments."""
    print("\n" + "=" * 90)
    print("GROUP 1 — SYSTEM EXPERIMENT RESULTS")
    print("Design matrix: chunking strategy × OCR × BM25")
    print("=" * 90)
    header = (f"{'Exp':<5} {'Label':<50} {'MRR':>7} {'R@5':>7} "
              f"{'R@10':>7} {'Judge':>7} {'N_ann':>6}")
    print(header)
    print("-" * 90)
    for m in all_metrics:
        if m.get("group") != 1:
            continue
        judge = (f"{m['LLM_judge']:.3f}"
                 if isinstance(m["LLM_judge"], float) else "N/A")
        print(
            f"{m['experiment']:<5} {m['label']:<50} "
            f"{m['MRR']:>7.4f} {m['Recall@5']:>7.4f} {m['Recall@10']:>7.4f} "
            f"{judge:>7} {m['annotated_n']:>6}"
        )
    print("=" * 90)

    # Summary: expected trends to verify isolation is working
    print("\nExpected trends (verify isolation is correct):")
    print("  OCR contribution  : E7 < E8  /  E4 < E5  /  E1 < E2")
    print("  BM25 contribution : E8 < E9  /  E5 < E6  /  E2 < E3")
    print("  C3 best strategy  : E7 ≥ E4 ≥ E1  (transcript only)")

    print("\nPer query-type MRR (Group 1):")
    print(f"{'Exp':<5} {'Conceptual':>12} {'Procedural':>12} {'Code':>8}")
    print("-" * 42)
    for m in all_metrics:
        if m.get("group") != 1:
            continue
        print(f"{m['experiment']:<5} {m['MRR_conceptual']:>12.4f} "
              f"{m['MRR_procedural']:>12.4f} {m['MRR_code']:>8.4f}")


def print_group2_table(all_metrics: list[dict]) -> None:
    """Prints the chunk parameter sensitivity table for Group 2."""
    print("\n" + "=" * 90)
    print("GROUP 2 — CHUNK PARAMETER SENSITIVITY")
    print("OCR: DISABLED | BM25: DISABLED  (pure chunking effect)")
    print("=" * 90)
    header = (f"{'Variant':<10} {'Type':<5} {'Param':<8} "
              f"{'MRR':>7} {'R@5':>7} {'R@10':>7} {'Judge':>7}")
    print(header)
    print("-" * 55)
    for m in all_metrics:
        if m.get("group") != 2:
            continue
        judge = (f"{m['LLM_judge']:.3f}"
                 if isinstance(m["LLM_judge"], float) else "N/A")
        chunk_type = m.get("chunk_type", "")
        parameter  = m.get("parameter", "")
        print(
            f"{m['experiment']:<10} {chunk_type:<5} {parameter:<8} "
            f"{m['MRR']:>7.4f} {m['Recall@5']:>7.4f} {m['Recall@10']:>7.4f} "
            f"{judge:>7}"
        )
    print("=" * 90)
    print("\nExpected: MRR should vary across chunk sizes/thresholds.")
    print("Optimal parameter → use that value for E4-E6 / E7-E9 family.")


def print_ablation_table(ablation_results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("ABLATION STUDY — Dataset statistics")
    print("=" * 80)
    print(f"{'Variant':<18} {'N_seg':>7} {'Avg_dur':>8} {'Avg_wds':>8} "
          f"{'OCR_fail%':>10} {'Code%':>7}")
    print("-" * 65)
    for r in ablation_results:
        if r.get("status") == "missing":
            print(f"{r['label']:<18}  FILE NOT FOUND")
            continue
        print(
            f"{r['label']:<18} {r['n_segments']:>7,} {r['avg_dur_sec']:>7.1f}s "
            f"{r['avg_words']:>8.1f} {r['ocr_fail_pct']:>9.1f}% {r['code_pct']:>6.1f}%"
        )
    print("=" * 80)

    print("\nPER-COURSE OCR FAILURE RATE (%):")
    courses = ["dsa", "daa", "dl", "os", "dbms", "cv", "coa", "ml", "cn"]
    print(f"{'Variant':<18} " + " ".join(f"{c:>6}" for c in courses))
    print("-" * 80)
    for r in ablation_results:
        if not r.get("by_course"):
            continue
        row = f"{r['label']:<18} "
        for c in courses:
            pct = r["by_course"].get(c, {}).get("ocr_fail_pct", "-")
            row += f"{str(pct):>6}"
        print(row)


# ─────────────────────────────────────────────────────────────────────────────
# CSV / JSON saving
# ─────────────────────────────────────────────────────────────────────────────

def save_csv_summary(
    group1_metrics:   list[dict],
    group2_metrics:   list[dict],
    ablation_results: list[dict],
) -> None:
    """Saves all results to eval_summary.csv for Excel import."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVAL_DIR / "eval_summary.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)

        # ── Group 1 ──────────────────────────────────────────────────────────
        writer.writerow(["GROUP 1 — SYSTEM EXPERIMENT RESULTS"])
        writer.writerow([
            "Experiment", "Label", "Strategy", "use_OCR", "use_BM25",
            "MRR", "Recall@5", "Recall@10", "LLM_Judge",
            "MRR_conceptual", "MRR_procedural", "MRR_code", "Annotated_N",
        ])
        for m in group1_metrics:
            writer.writerow([
                m["experiment"], m["label"],
                m.get("strategy", ""), m.get("use_ocr", ""), m.get("use_bm25", ""),
                m["MRR"], m["Recall@5"], m["Recall@10"], m["LLM_judge"],
                m["MRR_conceptual"], m["MRR_procedural"], m["MRR_code"],
                m["annotated_n"],
            ])

        writer.writerow([])

        # ── Group 2 ──────────────────────────────────────────────────────────
        writer.writerow(["GROUP 2 — CHUNK PARAMETER SENSITIVITY"])
        writer.writerow(["Note: OCR=OFF BM25=OFF for all Group 2 rows"])
        writer.writerow([
            "Variant", "Chunk_Type", "Parameter", "Strategy",
            "MRR", "Recall@5", "Recall@10", "LLM_Judge",
            "MRR_conceptual", "MRR_procedural", "MRR_code",
        ])
        for m in group2_metrics:
            writer.writerow([
                m["experiment"], m.get("chunk_type", ""), m.get("parameter", ""),
                m.get("strategy", ""),
                m["MRR"], m["Recall@5"], m["Recall@10"], m["LLM_judge"],
                m["MRR_conceptual"], m["MRR_procedural"], m["MRR_code"],
            ])

        writer.writerow([])

        # ── Ablation ─────────────────────────────────────────────────────────
        writer.writerow(["ABLATION STUDY — Dataset Statistics"])
        writer.writerow(["Variant", "N_segments", "Avg_dur_sec", "Avg_words",
                         "OCR_fail_pct", "Code_pct"])
        for r in ablation_results:
            if r.get("status") == "missing":
                writer.writerow([r["label"], "MISSING"])
                continue
            writer.writerow([
                r["label"], r["n_segments"], r["avg_dur_sec"],
                r["avg_words"], r["ocr_fail_pct"], r["code_pct"],
            ])

    print(f"\n  CSV saved → {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation framework for NPTEL lecture retrieval system"
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        choices=list(EXPERIMENTS.keys()),
        help="Run a single Group 1 experiment (E1–E9).",
    )
    parser.add_argument(
        "--sensitivity", type=str, default=None,
        choices=list(SENSITIVITY_VARIANTS.keys()),
        help="Run a single Group 2 sensitivity variant (e.g. C2-150).",
    )
    parser.add_argument(
        "--group1-only", action="store_true",
        help="Run Group 1 system experiments only (E1–E9).",
    )
    parser.add_argument(
        "--group2-only", action="store_true",
        help="Run Group 2 chunk sensitivity only.",
    )
    parser.add_argument(
        "--ablation-only", action="store_true",
        help="Run ablation dataset statistics only (no retrieval needed).",
    )
    parser.add_argument(
        "--no-llm-judge", action="store_true",
        help="Skip LLM judge scoring (faster).",
    )
    parser.add_argument(
        "--show-unfilled", action="store_true",
        help="List queries with expected_start_sec == 0.",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Retrieve top-k results per query (default 10).",
    )
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    use_llm_judge = not args.no_llm_judge

    # ── Show unfilled queries ─────────────────────────────────────────────────
    if args.show_unfilled:
        unfilled = [q for q in EVAL_QUERIES if q["expected_start_sec"] == 0]
        print(f"\n{len(unfilled)} queries need timestamp annotation:\n")
        for q in unfilled:
            print(f"  {q['id']:12} [{q['type']:12}] {q['expected_course']:6}  "
                  f"{q['query'][:55]}")
        print(f"\nEdit annotations.jsonl and fill expected_lecture "
              f"and expected_start_sec.\n")
        return

    # ── Ablation only ─────────────────────────────────────────────────────────
    if args.ablation_only:
        print("\nRunning ablation dataset analysis ...")
        ablation = run_ablation(EVAL_QUERIES, use_llm_judge=False)
        print_ablation_table(ablation)
        out = EVAL_DIR / "ablation_results.json"
        out.write_text(json.dumps(ablation, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"\n  Results saved → {out}")
        save_csv_summary([], [], ablation)
        return

    # ── Check retriever is importable before running anything ─────────────────
    try:
        import retriever  # noqa
    except ImportError as e:
        print(f"\n  Cannot import retriever.py: {e}")
        print("  Make sure retriever.py is in the same folder or on sys.path.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # Determine what to run
    # ─────────────────────────────────────────────────────────────────────────
    run_g1 = not args.group2_only
    run_g2 = not args.group1_only

    # Single experiment overrides
    if args.experiment:
        run_g1 = True
        run_g2 = False
    if args.sensitivity:
        run_g1 = False
        run_g2 = True

    # ─────────────────────────────────────────────────────────────────────────
    # Print evaluation set summary
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\nEvaluation set: {len(EVAL_QUERIES)} queries")
    annotated = sum(1 for q in EVAL_QUERIES if q["expected_start_sec"] > 0)
    print(f"Annotated with timestamps: {annotated}")
    if annotated == 0:
        print("  WARNING: No timestamps annotated. Metrics based on course-match only.")
        print("  Run --show-unfilled to see what needs filling.")

    group1_metrics  = []
    group2_metrics  = []
    judge_detail    = {}

    # ─────────────────────────────────────────────────────────────────────────
    # GROUP 1 — System Experiments (E1–E9)
    # ─────────────────────────────────────────────────────────────────────────
    if run_g1:
        exps_to_run = (
            {args.experiment: EXPERIMENTS[args.experiment]}
            if args.experiment
            else EXPERIMENTS
        )

        print(f"\n{'='*60}")
        print(f"GROUP 1: Running {len(exps_to_run)} system experiment(s)")
        print(f"{'='*60}")

        for exp_id, exp_config in exps_to_run.items():
            metrics = run_experiment(
                exp_id        = exp_id,
                exp_config    = exp_config,
                eval_queries  = EVAL_QUERIES,
                use_llm_judge = use_llm_judge,
                top_k         = args.top_k,
            )
            # Tag with group + chunk info for table printing
            metrics["group"]      = 1
            metrics["chunk_type"] = exp_config["strategy"].upper()
            group1_metrics.append(metrics)

            out_path = EVAL_DIR / f"results_{exp_id}.json"
            judge_detail[exp_id] = metrics.pop("judge_scores", [])
            out_path.write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"\n  {exp_id} done: MRR={metrics['MRR']:.4f} "
                  f"R@5={metrics['Recall@5']:.4f} "
                  f"Judge={metrics['LLM_judge']}")

        print_group1_table(group1_metrics)

    # ─────────────────────────────────────────────────────────────────────────
    # GROUP 2 — Chunk Parameter Sensitivity
    # ─────────────────────────────────────────────────────────────────────────
    if run_g2:
        variants_to_run = (
            {args.sensitivity: SENSITIVITY_VARIANTS[args.sensitivity]}
            if args.sensitivity
            else SENSITIVITY_VARIANTS
        )

        print(f"\n{'='*60}")
        print(f"GROUP 2: Running {len(variants_to_run)} sensitivity variant(s)")
        print("NOTE: OCR and BM25 are disabled for all Group 2 runs.")
        print(f"{'='*60}")

        for variant_id, variant_config in variants_to_run.items():
            # Safety assertion: group 2 must never enable OCR or BM25
            assert not variant_config["use_ocr"],  \
                f"Group 2 variant {variant_id} has use_ocr=True — must be False!"
            assert not variant_config["use_bm25"], \
                f"Group 2 variant {variant_id} has use_bm25=True — must be False!"

            metrics = run_experiment(
                exp_id        = variant_id,
                exp_config    = variant_config,
                eval_queries  = EVAL_QUERIES,
                use_llm_judge = use_llm_judge,
                top_k         = args.top_k,
            )
            # Tag with group + chunk info for table printing
            metrics["group"]      = 2
            metrics["chunk_type"] = variant_config["chunk_type"]
            metrics["parameter"]  = variant_config["parameter"]
            group2_metrics.append(metrics)

            # Save with a filename-safe variant ID
            safe_id  = variant_id.replace(".", "_").replace("-", "_")
            out_path = EVAL_DIR / f"sensitivity_{safe_id}.json"
            judge_detail[variant_id] = metrics.pop("judge_scores", [])
            out_path.write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"\n  {variant_id} done: MRR={metrics['MRR']:.4f} "
                  f"R@5={metrics['Recall@5']:.4f} "
                  f"Judge={metrics['LLM_judge']}")

        print_group2_table(group2_metrics)

    # ─────────────────────────────────────────────────────────────────────────
    # Ablation dataset statistics
    # ─────────────────────────────────────────────────────────────────────────
    # Run ablation stats when doing a full run (not single experiment/variant)
    if not args.experiment and not args.sensitivity:
        print("\n\nRunning ablation dataset analysis ...")
        ablation = run_ablation(EVAL_QUERIES, use_llm_judge=False)
        print_ablation_table(ablation)
        ablation_path = EVAL_DIR / "ablation_results.json"
        ablation_path.write_text(
            json.dumps(ablation, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        ablation = []

    # ─────────────────────────────────────────────────────────────────────────
    # Save all outputs
    # ─────────────────────────────────────────────────────────────────────────
    if judge_detail:
        judge_path = EVAL_DIR / "llm_judge_scores.json"
        judge_path.write_text(
            json.dumps(judge_detail, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    save_csv_summary(group1_metrics, group2_metrics, ablation)

    print(f"\n  All results saved to {EVAL_DIR}/")
    print("  Import eval_summary.csv into Excel to generate graphs.\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Final sanity check — warn if Group 1 results look suspiciously identical
    # ─────────────────────────────────────────────────────────────────────────
    if len(group1_metrics) >= 3:
        mrr_values = [m["MRR"] for m in group1_metrics]
        if len(set(mrr_values)) == 1:
            print("\n  ⚠️  WARNING: All Group 1 MRR values are identical!")
            print("     This suggests use_ocr/use_bm25 flags are still not being")
            print("     respected. Check that retriever.py accepts these parameters.")
        else:
            mrr_range = max(mrr_values) - min(mrr_values)
            print(f"\n  ✅  Group 1 MRR range: {min(mrr_values):.4f}–{max(mrr_values):.4f} "
                  f"(spread={mrr_range:.4f})")
            print("     Experiments are producing distinct results.")


if __name__ == "__main__":
    main()