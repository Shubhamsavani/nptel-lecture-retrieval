# NPTEL Lecture Retrieval

An AI-powered lecture retrieval system for NPTEL courses that enables efficient semantic search and retrieval over lecture transcripts and OCR-enhanced slide content.

---

## System Architecture

<p align="center">
  <img src="img/architecture.png" alt="System Architecture" width="850">
</p>

---

## Features

- Semantic lecture retrieval
- OCR-enhanced multimodal search
- Fast vector-based indexing using FAISS
- Hybrid retrieval with BM25 support
- Multiple chunking strategies (C1, C2, C3)
- Course-wise lecture organization
- Retrieval evaluation and benchmarking

---

## Tech Stack

- Python
- NLP / Machine Learning
- FAISS Vector Search
- Whisper ASR
- Tesseract OCR
- BM25 Retrieval
- Streamlit / Flask

---

## Project Structure

```bash
nptel-lecture-retrieval/
│
├── data/                  # Dataset and processed retrieval chunks
├── notebooks/             # Experiment notebooks
├── src/                   # Core retrieval pipeline
├── models/                # Embedding/index files
├── img/                   # Images and architecture diagrams
├── requirements.txt
├── app.py
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Shubhamsavani/nptel-lecture-retrieval.git
cd nptel-lecture-retrieval
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the application:

```bash
python app.py
```

Or launch notebooks:

```bash
jupyter notebook
```

---

## Retrieval Pipeline

1. Lecture videos are transcribed using Whisper ASR
2. Slides are processed using Tesseract OCR
3. Transcript and OCR text are fused
4. Content is chunked using multiple strategies
5. Embeddings are generated and indexed with FAISS
6. Queries are matched using semantic + BM25 retrieval

---

## Chunking Strategies

| Strategy | Description |
|---|---|
| C1 | Fixed time-based chunking |
| C2 | Utterance / fixed-word chunking |
| C3 | OCR Jaccard similarity slide-boundary chunking |

---

## Best Performing Configuration

| Setting | Value |
|---|---|
| Chunking | C3 Slide-Boundary |
| OCR | Enabled |
| BM25 | Enabled |
| MRR | 0.8259 |
| Recall@10 | 0.9643 |

---

## Future Improvements

- Cross-encoder reranking
- Multilingual retrieval
- Slide image embeddings
- Voice-based query support
- Advanced RAG pipelines

---

## Contributing

Contributions are welcome through pull requests and issue discussions.

---

## License

This project is licensed under the MIT License.
