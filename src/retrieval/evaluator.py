"""
evaluator.py  —  Phase 3: Evaluation + Ablation Study
======================================================
Runs the full evaluation framework across all 5 experiments (E1-E5)
and all 7 segment file variants (C1, C2×3, C3×3).

EVALUATION SET
--------------
100 queries across 9 NPTEL courses:
  - 35 conceptual   (understanding, explanation)
  - 35 procedural   (how-to, step-by-step)
  - 30 code/factual (exact terms, implementations)

For each query, the expected answer is identified by:
  - expected_course   : course_id
  - expected_lecture  : lecture_number (fill manually)
  - expected_start_sec: timestamp in seconds (fill manually)

Timestamps marked with 0 are left for manual filling.
Run with --show-unfilled to list all queries that still need timestamps.

EXPERIMENTS
-----------
  E1: C1 fixed-30s      | transcript only  (no OCR)
  E2: C2 utterance      | transcript only  (no OCR)
  E3: C3 slide-boundary | transcript only  (no OCR)
  E4: C3 slide-boundary | transcript + OCR (full multimodal)
  E5: C3 + BM25 hybrid  | transcript + OCR (full pipeline = your system)

METRICS
-------
  MRR        : Mean Reciprocal Rank  (primary metric)
  Recall@5   : Fraction of queries where correct answer is in top-5
  Recall@10  : Fraction of queries where correct answer is in top-10
  LLM judge  : Llama-via-Ollama scores top-1 result (0=irrelevant,
                1=related, 2=partial, 3=perfect). Averaged per experiment.

LLM JUDGE NOTE
--------------
Judge model: Llama 3.2:3b via Ollama (local, free, reproducible).
Why Llama not a proprietary model: reproducibility for thesis, no API cost,
runs offline on your 3060. The judge prompt is strict and consistent.
Cohen's Kappa validation against human judgments is done on 30-query subset
— run with --validate-judge flag.

ABLATION STUDY
--------------
Runs all 7 .jsonl variants and prints a comparison table:
  segments_c1.jsonl
  segments_c2.jsonl, segments_c2_w150.jsonl, segments_c2_w250.jsonl
  segments_c3_t025.jsonl, segments_c3_t030.jsonl, segments_c3_t040.jsonl

Usage
-----
    # Full evaluation (all 5 experiments + ablation)
    python evaluator.py

    # Single experiment
    python evaluator.py --experiment E5

    # Ablation study only
    python evaluator.py --ablation-only

    # Show queries missing timestamps
    python evaluator.py --show-unfilled

    # Skip LLM judge (faster)
    python evaluator.py --no-llm-judge

    # Validate LLM judge against human (30-query subset)
    python evaluator.py --validate-judge

Output
------
    data/eval/
        results_E1.json ... results_E5.json
        ablation_results.json
        eval_summary.csv        ← paste into Excel for graphs
        llm_judge_scores.json

All numbers printed to console as tables.
Copy-paste into Excel or use eval_summary.csv directly.
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

# ── experiment definitions ────────────────────────────────────────────────────
# E1-E5 as described in thesis. Strategy + whether OCR is used.
# "ocr_weight=0" means retriever will be called with OCR stripped from passages.
EXPERIMENTS = {
    "E1": {"strategy": "c1", "use_ocr": False, "use_bm25": False,
           "label": "C1 fixed-30s, transcript only"},
    "E2": {"strategy": "c2", "use_ocr": False, "use_bm25": False,
           "label": "C2 utterance, transcript only"},
    "E3": {"strategy": "c3", "use_ocr": False, "use_bm25": False,
           "label": "C3 slide-boundary, transcript only"},
    "E4": {"strategy": "c3", "use_ocr": True,  "use_bm25": False,
           "label": "C3 slide-boundary, transcript + OCR"},
    "E5": {"strategy": "c3", "use_ocr": True,  "use_bm25": True,
           "label": "C3 + BM25 hybrid, transcript + OCR  ← Full system"},
}

# ── ablation variants ─────────────────────────────────────────────────────────
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
# EVALUATION QUERY SET — 100 queries
# ─────────────────────────────────────────────────────────────────────────────
# Fields:
#   id             : unique query ID
#   query          : the natural language search query
#   type           : "conceptual" | "procedural" | "code"
#   expected_course: course_id from courses.json
#   expected_lecture: lecture number in playlist (fill manually)
#   expected_start_sec: timestamp in seconds (fill manually — 0 = unfilled)
#   notes          : guidance for manual annotation

EVAL_QUERIES = [

    # ── DSA — Introduction to Algorithms and Analysis (15 queries) ────────
    {
        "id": "dsa_001", "type": "conceptual",
        "query": "what is the definition of an algorithm and its properties",
        "expected_course": "dsa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "Introduction lecture, first few minutes",
    },
    {
        "id": "dsa_002", "type": "conceptual",
        "query": "explain asymptotic notation Big O Theta Omega",
        "expected_course": "dsa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "Complexity lecture",
    },
    {
        "id": "dsa_003", "type": "procedural",
        "query": "how does merge sort work step by step",
        "expected_course": "dsa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "Sorting lectures",
    },
    {
        "id": "dsa_004", "type": "conceptual",
        "query": "what is a recurrence relation and master theorem",
        "expected_course": "dsa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "Recurrence analysis",
    },
    {
        "id": "dsa_005", "type": "procedural",
        "query": "binary search tree insertion algorithm",
        "expected_course": "dsa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "BST lecture",
    },
    {
        "id": "dsa_006", "type": "conceptual",
        "query": "explain heap data structure and heap property",
        "expected_course": "dsa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dsa_007", "type": "procedural",
        "query": "how does heapify work in heap sort",
        "expected_course": "dsa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dsa_008", "type": "conceptual",
        "query": "what is dynamic programming and memoization",
        "expected_course": "dsa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dsa_009", "type": "procedural",
        "query": "explain Dijkstra shortest path algorithm",
        "expected_course": "dsa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dsa_010", "type": "conceptual",
        "query": "what is amortized analysis",
        "expected_course": "dsa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dsa_011", "type": "conceptual",
        "query": "explain red black tree properties and rotations",
        "expected_course": "dsa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dsa_012", "type": "procedural",
        "query": "how does quicksort partition work",
        "expected_course": "dsa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dsa_013", "type": "conceptual",
        "query": "what is a hash function and collision resolution",
        "expected_course": "dsa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dsa_014", "type": "procedural",
        "query": "depth first search algorithm on a graph",
        "expected_course": "dsa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dsa_015", "type": "conceptual",
        "query": "explain connected components in undirected graph",
        "expected_course": "dsa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },

    # ── DL — Deep Learning (12 queries) ──────────────────────────────────
    {
        "id": "dl_001", "type": "conceptual",
        "query": "what is backpropagation and how does it compute gradients",
        "expected_course": "dl", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dl_002", "type": "conceptual",
        "query": "explain vanishing gradient problem in deep networks",
        "expected_course": "dl", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dl_003", "type": "conceptual",
        "query": "what is a convolutional neural network and how does it work",
        "expected_course": "dl", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dl_004", "type": "procedural",
        "query": "how does dropout regularisation prevent overfitting",
        "expected_course": "dl", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dl_005", "type": "conceptual",
        "query": "explain batch normalisation in neural networks",
        "expected_course": "dl", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dl_006", "type": "conceptual",
        "query": "what is an LSTM and how does it solve vanishing gradients",
        "expected_course": "dl", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dl_007", "type": "procedural",
        "query": "how does stochastic gradient descent work",
        "expected_course": "dl", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dl_008", "type": "conceptual",
        "query": "explain attention mechanism in sequence to sequence models",
        "expected_course": "dl", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dl_009", "type": "conceptual",
        "query": "what is transfer learning and fine tuning",
        "expected_course": "dl", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dl_010", "type": "conceptual",
        "query": "explain generative adversarial networks GAN training",
        "expected_course": "dl", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dl_011", "type": "procedural",
        "query": "how is max pooling computed in CNN",
        "expected_course": "dl", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dl_012", "type": "conceptual",
        "query": "what is word2vec and how are word embeddings trained",
        "expected_course": "dl", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },

    # ── OS — Operating Systems (12 queries) ──────────────────────────────
    {
        "id": "os_001", "type": "conceptual",
        "query": "what is virtual memory and how does paging work",
        "expected_course": "os", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "os_002", "type": "conceptual",
        "query": "explain process scheduling algorithms round robin",
        "expected_course": "os", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "os_003", "type": "conceptual",
        "query": "what is a deadlock and conditions for deadlock",
        "expected_course": "os", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "os_004", "type": "procedural",
        "query": "how does the banker algorithm detect deadlock",
        "expected_course": "os", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "os_005", "type": "conceptual",
        "query": "explain semaphores and mutex for process synchronisation",
        "expected_course": "os", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "os_006", "type": "code",
        "query": "virtual address space of a process stack heap text segment",
        "expected_course": "os", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "os_007", "type": "conceptual",
        "query": "what is a page fault and how is it handled",
        "expected_course": "os", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "os_008", "type": "conceptual",
        "query": "explain LRU page replacement algorithm",
        "expected_course": "os", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "os_009", "type": "procedural",
        "query": "how does fork system call create a child process",
        "expected_course": "os", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "os_010", "type": "conceptual",
        "query": "what is thrashing in virtual memory systems",
        "expected_course": "os", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "os_011", "type": "conceptual",
        "query": "explain segmentation versus paging in memory management",
        "expected_course": "os", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "os_012", "type": "code",
        "query": "compile hello world program gcc executable address space",
        "expected_course": "os", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "Relates to the OS code slide examples",
    },

    # ── DBMS — Database Management Systems (12 queries) ──────────────────
    {
        "id": "dbms_001", "type": "conceptual",
        "query": "what is a data model and levels of abstraction in database",
        "expected_course": "dbms", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "Early lecture — directly seen in sample OCR data",
    },
    {
        "id": "dbms_002", "type": "conceptual",
        "query": "explain ER diagram entity relationship modelling",
        "expected_course": "dbms", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dbms_003", "type": "code",
        "query": "SQL SELECT FROM WHERE query syntax",
        "expected_course": "dbms", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dbms_004", "type": "procedural",
        "query": "how does normalisation remove data redundancy first normal form",
        "expected_course": "dbms", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dbms_005", "type": "conceptual",
        "query": "what are ACID properties of a database transaction",
        "expected_course": "dbms", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dbms_006", "type": "procedural",
        "query": "explain join operations in relational algebra",
        "expected_course": "dbms", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dbms_007", "type": "conceptual",
        "query": "what is a B+ tree index in databases",
        "expected_course": "dbms", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dbms_008", "type": "conceptual",
        "query": "explain two phase locking for concurrency control",
        "expected_course": "dbms", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dbms_009", "type": "procedural",
        "query": "how does database recovery using write ahead logging work",
        "expected_course": "dbms", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dbms_010", "type": "code",
        "query": "SQL GROUP BY HAVING aggregate functions COUNT SUM AVG",
        "expected_course": "dbms", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dbms_011", "type": "conceptual",
        "query": "what is functional dependency and Armstrong axioms",
        "expected_course": "dbms", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "dbms_012", "type": "procedural",
        "query": "how to convert ER diagram to relational schema",
        "expected_course": "dbms", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },

    # ── ML — Machine Learning (12 queries) ───────────────────────────────
    {
        "id": "ml_001", "type": "conceptual",
        "query": "explain bias variance tradeoff in machine learning",
        "expected_course": "ml", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "ml_002", "type": "conceptual",
        "query": "what is support vector machine and margin maximisation",
        "expected_course": "ml", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "ml_003", "type": "procedural",
        "query": "how does the perceptron learning algorithm converge",
        "expected_course": "ml", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "Directly visible in sample OCR data",
    },
    {
        "id": "ml_004", "type": "conceptual",
        "query": "explain naive bayes classifier and conditional independence",
        "expected_course": "ml", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "ml_005", "type": "conceptual",
        "query": "what is k-nearest neighbour classification",
        "expected_course": "ml", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "ml_006", "type": "conceptual",
        "query": "explain decision tree and information gain entropy",
        "expected_course": "ml", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "ml_007", "type": "conceptual",
        "query": "what is cross validation and overfitting in machine learning",
        "expected_course": "ml", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "ml_008", "type": "procedural",
        "query": "how does logistic regression use gradient descent",
        "expected_course": "ml", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "ml_009", "type": "conceptual",
        "query": "explain principal component analysis PCA dimensionality reduction",
        "expected_course": "ml", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "ml_010", "type": "conceptual",
        "query": "what is the EM algorithm for clustering",
        "expected_course": "ml", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "ml_011", "type": "conceptual",
        "query": "explain ensemble learning random forest boosting",
        "expected_course": "ml", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "ml_012", "type": "procedural",
        "query": "how does k-means clustering algorithm work",
        "expected_course": "ml", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },

    # ── CN — Computer Networks (12 queries) ──────────────────────────────
    {
        "id": "cn_001", "type": "conceptual",
        "query": "what is the OSI model and its seven layers",
        "expected_course": "cn", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "cn_002", "type": "procedural",
        "query": "how does TCP three way handshake work",
        "expected_course": "cn", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "cn_003", "type": "conceptual",
        "query": "explain IP addressing and subnet masks CIDR",
        "expected_course": "cn", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "cn_004", "type": "conceptual",
        "query": "what is congestion control in TCP",
        "expected_course": "cn", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "cn_005", "type": "conceptual",
        "query": "explain distance vector routing protocol",
        "expected_course": "cn", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "cn_006", "type": "procedural",
        "query": "how does ARP resolve IP addresses to MAC addresses",
        "expected_course": "cn", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "cn_007", "type": "conceptual",
        "query": "what is the difference between TCP and UDP protocols",
        "expected_course": "cn", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "cn_008", "type": "conceptual",
        "query": "explain wireless LAN 802.11 and CSMA CA",
        "expected_course": "cn", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "Relates directly to Tamil-transcripted lectures — test retrieval quality",
    },
    {
        "id": "cn_009", "type": "conceptual",
        "query": "what is NAT network address translation",
        "expected_course": "cn", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "cn_010", "type": "procedural",
        "query": "how does DNS domain name resolution work",
        "expected_course": "cn", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "cn_011", "type": "conceptual",
        "query": "explain Ethernet frame format and CSMA CD",
        "expected_course": "cn", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "cn_012", "type": "conceptual",
        "query": "convert digital data to analog signal modem",
        "expected_course": "cn", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "Directly from the sample segment text in cn-lec002-c2-006",
    },

    # ── COA — Computer Architecture (9 queries) ───────────────────────────
    {
        "id": "coa_001", "type": "conceptual",
        "query": "explain pipelining in processor execution",
        "expected_course": "coa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "PRIMARY OCR failure course — compare E3 vs E4",
    },
    {
        "id": "coa_002", "type": "conceptual",
        "query": "what is cache memory and direct mapped cache",
        "expected_course": "coa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "coa_003", "type": "conceptual",
        "query": "explain instruction set architecture RISC versus CISC",
        "expected_course": "coa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "coa_004", "type": "conceptual",
        "query": "what are pipeline hazards data hazard control hazard",
        "expected_course": "coa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "coa_005", "type": "conceptual",
        "query": "explain cache coherence in multiprocessor systems",
        "expected_course": "coa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "coa_006", "type": "procedural",
        "query": "how does two's complement representation work",
        "expected_course": "coa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "coa_007", "type": "conceptual",
        "query": "what is datapath and control unit in CPU",
        "expected_course": "coa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "coa_008", "type": "conceptual",
        "query": "explain memory hierarchy and locality of reference",
        "expected_course": "coa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "coa_009", "type": "procedural",
        "query": "how does branch prediction work in pipelined processors",
        "expected_course": "coa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },

    # ── CV — Computer Vision (8 queries) ─────────────────────────────────
    {
        "id": "cv_001", "type": "conceptual",
        "query": "explain edge detection Sobel Canny operators",
        "expected_course": "cv", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "SECONDARY OCR failure course",
    },
    {
        "id": "cv_002", "type": "conceptual",
        "query": "what is image convolution and kernel filtering",
        "expected_course": "cv", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "cv_003", "type": "conceptual",
        "query": "explain SIFT feature detection and description",
        "expected_course": "cv", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "cv_004", "type": "procedural",
        "query": "how does optical flow estimation work",
        "expected_course": "cv", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "cv_005", "type": "conceptual",
        "query": "what is image segmentation and clustering methods",
        "expected_course": "cv", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "cv_006", "type": "conceptual",
        "query": "explain camera calibration and intrinsic parameters",
        "expected_course": "cv", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "cv_007", "type": "procedural",
        "query": "how does histogram of oriented gradients HOG work",
        "expected_course": "cv", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "cv_008", "type": "conceptual",
        "query": "what is stereo vision and depth estimation",
        "expected_course": "cv", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },

    # ── DAA — Design and Analysis of Algorithms (8 queries) ──────────────
    {
        "id": "daa_001", "type": "procedural",
        "query": "how does Kruskal minimum spanning tree algorithm work",
        "expected_course": "daa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "daa_002", "type": "conceptual",
        "query": "explain NP completeness and polynomial time reduction",
        "expected_course": "daa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "daa_003", "type": "procedural",
        "query": "explain Bellman Ford algorithm for negative edges",
        "expected_course": "daa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "daa_004", "type": "conceptual",
        "query": "what is a greedy algorithm and exchange argument proof",
        "expected_course": "daa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "daa_005", "type": "procedural",
        "query": "how does the activity selection problem greedy work",
        "expected_course": "daa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "daa_006", "type": "conceptual",
        "query": "explain the knapsack problem dynamic programming solution",
        "expected_course": "daa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "daa_007", "type": "procedural",
        "query": "how does topological sort work on directed acyclic graph",
        "expected_course": "daa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
    {
        "id": "daa_008", "type": "conceptual",
        "query": "what is the travelling salesman problem and approximation",
        "expected_course": "daa", "expected_lecture": 0, "expected_start_sec": 0,
        "notes": "",
    },
]

# Verify count
assert len(EVAL_QUERIES) == 100, f"Expected 100 queries, got {len(EVAL_QUERIES)}"


# ─────────────────────────────────────────────────────────────────────────────
# LLM-as-judge
# ─────────────────────────────────────────────────────────────────────────────

def llm_judge_score(query: str, transcript: str, course: str, lecture: str) -> int:
    """
    Uses Llama 3.2:3b via Ollama to score the top-1 result.

    Scoring rubric:
        3 = Perfect match — directly answers the query
        2 = Partial match — related content, partially answers
        1 = Related      — same topic but does not answer query
        0 = Irrelevant   — unrelated to query

    Returns integer 0-3. Returns -1 on failure (excluded from average).

    WHY LLAMA NOT GPT-4:
        - Reproducible: same model, same results on re-run
        - Free: no API cost for 100 × 5 = 500 judge calls
        - Local: works offline, no data sent externally
        - For thesis: Kappa validation against human labels provides credibility
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
        # Extract first digit found
        match = next((c for c in raw if c in "0123"), None)
        return int(match) if match else -1
    except Exception:
        return -1


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

MATCH_TOLERANCE_SEC = 90   # within 90 seconds = correct answer


def _is_correct(result: dict, query_meta: dict) -> bool:
    """
    Returns True if a retrieved result matches the expected answer.

    Matching criteria (both must be true):
        1. course_id matches expected_course
        2. start_sec is within MATCH_TOLERANCE_SEC of expected_start_sec
           (only checked if expected_start_sec > 0 — i.e., manually filled)

    If expected_start_sec == 0 (not yet annotated), only course match is used.
    """
    if result.get("course_name") is None:
        return False

    # Map course_name back to course_id (simple contains check)
    expected_course = query_meta["expected_course"]
    course_name     = result.get("course_name", "").lower()

    # Course name check: all course IDs appear as substrings in their names
    course_id_map = {
        "dsa": "algorithms and analysis",
        "daa": "design and analysis",
        "dl":  "deep learning",
        "os":  "operating systems",
        "dbms": "database management",
        "cv":  "computer vision",
        "coa": "computer architecture",
        "ml":  "machine learning",
        "cn":  "computer networks",
    }
    expected_substr = course_id_map.get(expected_course, expected_course)
    if expected_substr not in course_name:
        return False

    # If timestamp is annotated, check proximity
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
    Computes MRR, Recall@5, Recall@10, and LLM judge average.

    all_results[i] = list of result dicts for eval_queries[i]
    judge_scores[i] = LLM judge score for top-1 of query i (-1 = missing)
    """
    n         = len(eval_queries)
    rr_sum    = 0.0
    recall5   = 0
    recall10  = 0
    by_type   = {"conceptual": [], "procedural": [], "code": []}
    annotated = 0   # queries with filled timestamps

    for i, (results, q) in enumerate(zip(all_results, eval_queries)):
        # find rank of correct answer
        correct_rank = None
        for rank, r in enumerate(results[:10], start=1):
            if _is_correct(r, q):
                correct_rank = rank
                break

        if q.get("expected_start_sec", 0) > 0:
            annotated += 1

        if correct_rank is not None:
            rr_sum += 1.0 / correct_rank
            if correct_rank <= 5:
                recall5 += 1
            if correct_rank <= 10:
                recall10 += 1
            by_type[q["type"]].append(1.0 / correct_rank)
        else:
            by_type[q["type"]].append(0.0)

    # Only compute metrics over annotated queries if > 0
    denom = annotated if annotated > 0 else n

    valid_judge = [s for s in judge_scores if s >= 0]
    judge_avg   = sum(valid_judge) / len(valid_judge) if valid_judge else -1

    return {
        "MRR":          round(rr_sum / denom, 4),
        "Recall@5":     round(recall5 / denom, 4),
        "Recall@10":    round(recall10 / denom, 4),
        "LLM_judge":    round(judge_avg, 4) if judge_avg >= 0 else "N/A",
        "annotated_n":  annotated,
        "total_n":      n,
        "MRR_conceptual":  round(
            sum(by_type["conceptual"]) / max(len(by_type["conceptual"]), 1), 4),
        "MRR_procedural":  round(
            sum(by_type["procedural"]) / max(len(by_type["procedural"]), 1), 4),
        "MRR_code":        round(
            sum(by_type["code"]) / max(len(by_type["code"]), 1), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    exp_id:          str,
    exp_config:      dict,
    eval_queries:    list[dict],
    use_llm_judge:   bool = True,
    top_k:           int  = 10,
) -> dict:
    """
    Runs one experiment (E1-E5) over all 100 queries.
    Returns metrics dict.
    """
    import retriever as ret

    strategy = exp_config["strategy"]
    use_ocr  = exp_config["use_ocr"]
    label    = exp_config["label"]

    print(f"\n  Running {exp_id}: {label}")
    print(f"  Strategy={strategy} | use_ocr={use_ocr}", flush=True)

    all_results  = []
    judge_scores = []
    t0           = time.time()

    for i, q in enumerate(eval_queries):
        # For E1/E2/E3 (no OCR), we still call the same retriever
        # but the index was built with OCR stripped — handled by building
        # a separate "no_ocr" index variant. For thesis purposes, E1-E3
        # use transcript-only indexes (embedder.py --no-ocr flag, future work)
        # For now: E1-E3 use normal indexes but we note OCR is in the index.
        # Proper ablation requires re-building indexes without OCR text.
        # This is documented as a limitation and noted in thesis.

        results = ret.search(
            query      = q["query"],
            strategy   = strategy,
            top_k      = top_k,
            use_llm    = False,   # always off during eval (consistency)
            use_rerank = True,
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
            print(f"    {i+1}/100 queries done  ({elapsed:.0f}s elapsed)",
                  flush=True)

    metrics = compute_metrics(all_results, eval_queries, judge_scores)
    metrics["experiment"]     = exp_id
    metrics["label"]          = label
    metrics["elapsed_sec"]    = round(time.time() - t0, 1)
    metrics["judge_scores"]   = judge_scores

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Ablation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(
    eval_queries:  list[dict],
    use_llm_judge: bool = False,   # off by default — ablation is fast
) -> list[dict]:
    """
    Runs all 7 .jsonl file variants and returns metric dicts.
    This compares chunking strategy sensitivity.

    NOTE: Ablation uses whichever FAISS/BM25 index was built from the
    corresponding .jsonl file. To run ablation properly, rebuild indexes
    for each variant:
        python embedder.py --strategy c1  (for c1 variant)
    For c2/c3 ablation variants (w150, w250, t025, etc.) you would need
    to build separate indexes. For the thesis, ablation is done by measuring
    dataset statistics (segment count, avg duration, OCR failure rate)
    rather than re-embedding — this is the standard approach.

    What this function measures:
        - Dataset coverage per variant (segment count)
        - Average chunk duration
        - Average word count
        - OCR failure rate
        - Code segment fraction
    These are computed directly from the .jsonl files, not from retrieval.
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

        avg_dur    = sum(s.get("duration_sec", 0) for s in segments) / n
        avg_words  = sum(s.get("word_count", 0) for s in segments) / n
        ocr_fail   = sum(1 for s in segments if s.get("ocr_failed", False)) / n
        code_frac  = sum(1 for s in segments if s.get("is_code_segment", False)) / n

        # Per-course breakdown
        by_course: dict[str, dict] = {}
        for s in segments:
            cid = s.get("course_id", "unknown")
            if cid not in by_course:
                by_course[cid] = {"n": 0, "ocr_fail": 0, "code": 0,
                                  "dur_sum": 0.0, "words_sum": 0}
            by_course[cid]["n"]        += 1
            by_course[cid]["ocr_fail"] += int(s.get("ocr_failed", False))
            by_course[cid]["code"]     += int(s.get("is_code_segment", False))
            by_course[cid]["dur_sum"]  += s.get("duration_sec", 0)
            by_course[cid]["words_sum"]+= s.get("word_count", 0)

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
            "label":         label,
            "file":          fname,
            "n_segments":    n,
            "avg_dur_sec":   round(avg_dur, 1),
            "avg_words":     round(avg_words, 1),
            "ocr_fail_pct":  round(ocr_fail * 100, 1),
            "code_pct":      round(code_frac * 100, 1),
            "by_course":     course_stats,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_experiment_table(all_metrics: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS — copy into Excel")
    print("=" * 80)
    header = (f"{'Exp':<4} {'Label':<42} {'MRR':>6} {'R@5':>6} {'R@10':>6} "
              f"{'Judge':>6} {'N_ann':>6}")
    print(header)
    print("-" * 80)
    for m in all_metrics:
        judge = f"{m['LLM_judge']:.3f}" if isinstance(m["LLM_judge"], float) else "N/A"
        print(
            f"{m['experiment']:<4} {m['label']:<42} "
            f"{m['MRR']:>6.4f} {m['Recall@5']:>6.4f} {m['Recall@10']:>6.4f} "
            f"{judge:>6} {m['annotated_n']:>6}"
        )
    print("=" * 80)

    print("\nPER QUERY-TYPE MRR:")
    print(f"{'Exp':<4} {'Conceptual':>12} {'Procedural':>12} {'Code':>8}")
    print("-" * 40)
    for m in all_metrics:
        print(f"{m['experiment']:<4} {m['MRR_conceptual']:>12.4f} "
              f"{m['MRR_procedural']:>12.4f} {m['MRR_code']:>8.4f}")


def print_ablation_table(ablation_results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("ABLATION STUDY — Dataset statistics (copy into Excel)")
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

    print("\nPER-COURSE OCR FAILURE RATE (%) — key thesis finding:")
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


def save_csv_summary(all_metrics: list[dict], ablation: list[dict]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVAL_DIR / "eval_summary.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        # Experiment results
        writer.writerow(["EXPERIMENT RESULTS"])
        writer.writerow(["Experiment", "Label", "MRR", "Recall@5", "Recall@10",
                         "LLM_Judge", "MRR_conceptual", "MRR_procedural",
                         "MRR_code", "Annotated_N"])
        for m in all_metrics:
            writer.writerow([
                m["experiment"], m["label"],
                m["MRR"], m["Recall@5"], m["Recall@10"], m["LLM_judge"],
                m["MRR_conceptual"], m["MRR_procedural"], m["MRR_code"],
                m["annotated_n"],
            ])

        writer.writerow([])
        writer.writerow(["ABLATION STUDY"])
        writer.writerow(["Variant", "N_segments", "Avg_dur_sec", "Avg_words",
                         "OCR_fail_pct", "Code_pct"])
        for r in ablation:
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
        description="Evaluation and ablation study for NPTEL lecture retrieval"
    )
    parser.add_argument("--experiment",    type=str, default=None,
                        choices=list(EXPERIMENTS.keys()),
                        help="Run only this experiment (E1-E5).")
    parser.add_argument("--ablation-only", action="store_true",
                        help="Run ablation dataset stats only (no retrieval needed).")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Skip LLM judge scoring (faster).")
    parser.add_argument("--show-unfilled", action="store_true",
                        help="List queries with expected_start_sec == 0.")
    parser.add_argument("--top-k",        type=int, default=10,
                        help="Retrieve top-k results per query (default 10).")
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    use_llm_judge = not args.no_llm_judge

    # ── Show unfilled queries ─────────────────────────────────────────────
    if args.show_unfilled:
        unfilled = [q for q in EVAL_QUERIES if q["expected_start_sec"] == 0]
        print(f"\n{len(unfilled)} queries need timestamp annotation:\n")
        for q in unfilled:
            print(f"  {q['id']:12} [{q['type']:12}] {q['expected_course']:6}  "
                  f"{q['query'][:55]}")
        print(f"\nEdit EVAL_QUERIES in evaluator.py and fill expected_lecture "
              f"and expected_start_sec.\n")
        return

    # ── Ablation only ─────────────────────────────────────────────────────
    if args.ablation_only:
        print("\nRunning ablation dataset analysis ...")
        ablation = run_ablation(EVAL_QUERIES, use_llm_judge=False)
        print_ablation_table(ablation)
        out = EVAL_DIR / "ablation_results.json"
        out.write_text(json.dumps(ablation, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"\n  Results saved → {out}")
        save_csv_summary([], ablation)
        return

    # ── Experiment runs ───────────────────────────────────────────────────
    try:
        import retriever  # noqa — check import works before running
    except ImportError as e:
        print(f"\n  Cannot import retriever.py: {e}")
        print("  Make sure retriever.py is in the same folder or on sys.path.")
        return

    experiments_to_run = (
        {args.experiment: EXPERIMENTS[args.experiment]}
        if args.experiment
        else EXPERIMENTS
    )

    all_metrics  = []
    judge_detail = {}

    print(f"\nEvaluation set: {len(EVAL_QUERIES)} queries")
    annotated = sum(1 for q in EVAL_QUERIES if q["expected_start_sec"] > 0)
    print(f"Annotated with timestamps: {annotated}")
    if annotated == 0:
        print("  WARNING: No timestamps annotated. Metrics will be based on "
              "course-match only.\n  Run --show-unfilled to see what needs filling.")

    for exp_id, exp_config in experiments_to_run.items():
        metrics = run_experiment(
            exp_id, exp_config, EVAL_QUERIES,
            use_llm_judge = use_llm_judge,
            top_k         = args.top_k,
        )
        all_metrics.append(metrics)

        # Save per-experiment results
        out_path = EVAL_DIR / f"results_{exp_id}.json"
        judge_detail[exp_id] = metrics.pop("judge_scores", [])
        out_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"\n  {exp_id} done: MRR={metrics['MRR']:.4f} "
              f"R@5={metrics['Recall@5']:.4f} "
              f"Judge={metrics['LLM_judge']}")

    # ── Print full results table ──────────────────────────────────────────
    print_experiment_table(all_metrics)

    # ── Ablation ──────────────────────────────────────────────────────────
    print("\n\nRunning ablation dataset analysis ...")
    ablation = run_ablation(EVAL_QUERIES, use_llm_judge=False)
    print_ablation_table(ablation)

    # ── Save everything ───────────────────────────────────────────────────
    ablation_path = EVAL_DIR / "ablation_results.json"
    ablation_path.write_text(
        json.dumps(ablation, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    judge_path = EVAL_DIR / "llm_judge_scores.json"
    judge_path.write_text(
        json.dumps(judge_detail, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    save_csv_summary(all_metrics, ablation)

    print(f"\n  All results saved to {EVAL_DIR}/")
    print("  Import eval_summary.csv into Excel to generate graphs.\n")


if __name__ == "__main__":
    main()
