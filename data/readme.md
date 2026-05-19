# Dataset Details

This folder contains the multimodal dataset used for the NPTEL Lecture Retrieval system.

The dataset was built using:
- Whisper ASR for lecture transcription
- Tesseract OCR for slide text extraction
- Multiple chunking strategies (C1, C2, C3)
- OCR + BM25 enhanced retrieval experiments

---

# Dataset Structure

```bash
data/
│
├── raw_transcripts/        # Whisper transcripts
├── ocr_text/               # OCR extracted slide text
├── processed_chunks/       # Final retrieval chunks
├── metadata/               # Course metadata
├── eval/                   # Evaluation queries and metrics
└── README.md
```

---

# Chunking Strategies

| Strategy | Description |
|---|---|
| C1 | Fixed 30-second chunks |
| C2 | Utterance / fixed-word chunking |
| C3 | OCR Jaccard similarity slide-boundary chunking |

---

# Best Performing Configuration

| Setting | Value |
|---|---|
| Chunking | C3 |
| OCR | Enabled |
| BM25 | Enabled |
| MRR | 0.8259 |
| Recall@10 | 0.9643 |

---

# Evaluation

The `eval/` folder contains:
- Retrieval experiment outputs
- Quantitative evaluation metrics
- `annotation.jsonl` containing 100 human-annotated evaluation queries with:
  - Expected course
  - Lecture number
  - Timestamp annotations
  - Query relevance labels

Example queries include:
- “what is backpropagation and how does it compute gradients”
- “explain pipelining in processor execution”
- “how does TCP three way handshake work” :contentReference[oaicite:0]{index=0}

For extending annotations and evaluating new retrieval systems, the repository includes `eval_app.py` — a custom UI where human evaluators can:
- Search queries
- Inspect retrieved results
- Select the most relevant lecture segment
- Generate updated evaluation annotations

Metrics used:
- MRR
- Recall@5
- Recall@10
- LLM relevance scoring

---

# Retrieval Experiment Results

## GROUP 1 — System Experiment Results

| Exp | Strategy | OCR | BM25 | MRR | Recall@5 | Recall@10 |
|---|---|---|---|---|---|---|
| E1 | C1 transcript only | ❌ | ❌ | 0.4478 | 0.5952 | 0.6786 |
| E2 | C1 + OCR | ✅ | ❌ | 0.6536 | 0.7857 | 0.8452 |
| E3 | C1 + OCR + BM25 | ✅ | ✅ | 0.6564 | 0.7976 | 0.8571 |
| E4 | C2 transcript only | ❌ | ❌ | 0.4059 | 0.5595 | 0.7024 |
| E5 | C2 + OCR | ✅ | ❌ | 0.6066 | 0.7262 | 0.7857 |
| E6 | C2 + OCR + BM25 | ✅ | ✅ | 0.6148 | 0.7619 | 0.8214 |
| E7 | C3 transcript only | ❌ | ❌ | 0.4487 | 0.6429 | 0.6786 |
| E8 | C3 + OCR | ✅ | ❌ | 0.8200 | 0.9643 | 0.9643 |
| E9 | C3 + OCR + BM25 (Full System) | ✅ | ✅ | **0.8259** | **0.9524** | **0.9643** |

---

# Key Findings

## Impact of OCR

Adding OCR consistently improves retrieval quality across all chunking strategies.

### Example
- C1 MRR:
  - Without OCR → 0.4478
  - With OCR → 0.6536

- C3 MRR:
  - Without OCR → 0.4487
  - With OCR → 0.8200

OCR provides major gains by incorporating slide-level semantic information.

---

# Impact of BM25

BM25 hybrid retrieval provides additional improvements after OCR fusion.

### Example
- C3 + OCR:
  - MRR = 0.8200

- C3 + OCR + BM25:
  - MRR = 0.8259

Dense + sparse hybrid retrieval improves ranking robustness.

---

# Best Performing Configuration

## Full System

### Configuration
- Chunking: C3 slide-boundary segmentation
- OCR: Enabled
- BM25: Enabled

### Performance
- MRR: 0.8259
- Recall@5: 0.9524
- Recall@10: 0.9643

This configuration achieved the best overall retrieval performance.

---

# GROUP 2 — Chunk Parameter Sensitivity

## C2 Word Chunking Sensitivity

| Variant | MRR | Recall@5 | Recall@10 |
|---|---|---|---|
| 150 words | 0.4527 | 0.6310 | 0.7381 |
| 200 words | 0.4059 | 0.5595 | 0.7024 |
| 250 words | 0.3762 | 0.5238 | 0.6429 |

### Observation
Smaller chunk sizes performed better for retrieval.

---

## C3 Threshold Sensitivity

| Threshold | MRR | Recall@5 | Recall@10 |
|---|---|---|---|
| t = 0.25 | 0.4494 | 0.6429 | 0.6905 |
| t = 0.30 | 0.4487 | 0.6429 | 0.6786 |
| t = 0.40 | 0.4467 | 0.6548 | 0.6905 |

### Observation
Threshold variations showed relatively stable performance.

---

# Dataset Statistics

| Variant | Segments | Avg Duration (sec) | Avg Words | OCR Failure % | Code % |
|---|---|---|---|---|---|
| C1 (30s) | 24,804 | 33.2 | 80.6 | 3.4 | 3.6 |
| C2 (150w) | 12,851 | 65.2 | 155.5 | 2.2 | 5.2 |
| C2 (200w) | 9,841 | 85.3 | 203.1 | 1.9 | 6.2 |
| C2 (250w) | 7,991 | 105.2 | 250.1 | 1.5 | 7.3 |
| C3 (0.25) | 15,265 | 54.4 | 130.9 | 1.8 | 3.3 |
| C3 (0.40) | 17,620 | 47.1 | 113.4 | 1.5 | 3.0 |

---

# Metadata Included

Each retrieval chunk may contain:

- Course Name
- Lecture ID
- Timestamp
- Transcript Text
- OCR Text
- Chunk ID
- Chunking Strategy
- Source Video Path
- Retrieval Metadata

---

# Research Objectives

This dataset supports research in:

- Lecture Retrieval
- Educational Search Systems
- Retrieval-Augmented Generation (RAG)
- Multimodal Information Retrieval
- OCR-Augmented Retrieval
- Timestamp-Aware Search
- Hybrid Dense + Sparse Retrieval

---

# Future Improvements

- Multilingual lecture support
- OCR correction pipelines
- Slide image embeddings
- Code-aware retrieval
- Query expansion
- Cross-encoder reranking
- Adaptive chunk merging

---

# Repository

Repository:
https://github.com/Shubhamsavani/nptel-lecture-retrieval
