"""
app.py  —  NPTEL Lecture Retrieval System — Streamlit UI
=========================================================
A clean, academic-grade search interface for the lecture retrieval system.

Features:
  - Natural language query input
  - Strategy selector (C1 / C2 / C3)
  - LLM toggle (requires Ollama running)
  - Re-ranking toggle
  - Result cards with deep-link YouTube buttons
  - Query intent display
  - Per-result expandable transcript + OCR text
  - Course filter sidebar
  - Query history

Running
-------
    streamlit run app.py

    # Custom port
    streamlit run app.py --server.port 8502

    # If app.py is not in the project root, set the path:
    streamlit run src/app.py

Environment variables (from .env):
    PROJECT_ROOT       — project root directory
    EMBEDDING_DEVICE   — "cuda" or "cpu"
    OLLAMA_MODEL       — ollama model name (default: llama3.2:3b)
    DEFAULT_STRATEGY   — default chunking strategy (default: c2)
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
# Allows importing retriever.py regardless of where app.py is placed
_here = Path(__file__).resolve().parent
for _candidate in [_here, _here / "src" / "retrieval", _here / "src"]:
    if (_candidate / "retriever.py").exists():
        sys.path.insert(0, str(_candidate))
        break

# ── load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    for _c in [_here, _here.parent]:
        if (_c / ".env").exists():
            load_dotenv(_c / ".env")
            break
except ImportError:
    pass

# ── page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="NPTEL Lecture Retrieval",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  /* Header */
  .app-header {
    padding: 1.5rem 0 1rem 0;
    border-bottom: 2px solid #1a1a2e;
    margin-bottom: 1.5rem;
  }
  .app-title {
    font-size: 1.8rem;
    font-weight: 600;
    color: #1a1a2e;
    letter-spacing: -0.5px;
    margin: 0;
  }
  .app-subtitle {
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* Result cards */
  .result-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #1a1a2e;
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    transition: box-shadow 0.2s;
  }
  .result-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  }
  .result-card.code-card { border-left-color: #f59e0b; }
  .result-card.theory-card { border-left-color: #6366f1; }
  .result-card.conceptual-card { border-left-color: #10b981; }

  .rank-badge {
    display: inline-block;
    background: #1a1a2e;
    color: white;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 3px;
    margin-right: 8px;
    font-weight: 500;
  }
  .course-tag {
    display: inline-block;
    background: #f3f4f6;
    color: #374151;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 12px;
    margin-right: 6px;
    font-weight: 500;
  }
  .type-tag {
    display: inline-block;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 12px;
    font-weight: 500;
  }
  .type-code       { background: #fef3c7; color: #92400e; }
  .type-theoretical { background: #ede9fe; color: #4c1d95; }
  .type-conceptual { background: #d1fae5; color: #065f46; }

  .lecture-title {
    font-size: 1rem;
    font-weight: 600;
    color: #111827;
    margin: 0.5rem 0 0.25rem 0;
  }
  .timestamp-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #6b7280;
    margin-bottom: 0.6rem;
  }
  .snippet {
    font-size: 0.87rem;
    color: #374151;
    line-height: 1.6;
    border-left: 2px solid #e5e7eb;
    padding-left: 10px;
    margin: 0.5rem 0;
    font-style: italic;
  }
  .play-btn {
    display: inline-block;
    background: #dc2626;
    color: white !important;
    text-decoration: none !important;
    padding: 6px 14px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 0.5rem;
    letter-spacing: 0.3px;
  }
  .play-btn:hover { background: #b91c1c; }

  .score-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #9ca3af;
    margin-top: 0.4rem;
  }

  /* Intent badge */
  .intent-badge {
    display: inline-block;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    margin-left: 8px;
  }
  .intent-code        { background: #fef3c7; color: #92400e; }
  .intent-theoretical { background: #ede9fe; color: #4c1d95; }
  .intent-conceptual  { background: #d1fae5; color: #065f46; }

  /* No results */
  .no-results {
    text-align: center;
    padding: 3rem;
    color: #9ca3af;
    font-size: 0.9rem;
  }

  /* Streamlit widget tweaks */
  .stTextInput > div > div > input {
    font-size: 1rem;
    border-radius: 6px;
  }
  div[data-testid="stExpander"] {
    border: 1px solid #e5e7eb;
    border-radius: 6px;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #f9fafb;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Retriever loader (cached so models load once)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading retrieval models…")
def load_retriever():
    """
    Imports retriever module and warms up the embedding model.
    @st.cache_resource ensures this runs exactly once per session.
    Returns the retriever module itself.
    """
    try:
        import retriever as _ret
        # Warm up: trigger model load now so first query is fast
        _ret._get_embed_model()
        return _ret
    except ImportError as e:
        st.error(
            f"Could not import retriever.py: {e}\n\n"
            "Make sure retriever.py is in the same folder as app.py "
            "or in src/retrieval/."
        )
        st.stop()
    except FileNotFoundError as e:
        st.error(
            f"Index file not found: {e}\n\n"
            "Run these commands first:\n"
            "  python embedder.py\n"
            "  python bm25_builder.py"
        )
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_LABELS = {
    "c1": "C1 — Fixed 30s window",
    "c2": "C2 — Utterance window (200 words)",
    "c3": "C3 — Slide boundary (OCR Jaccard)",
}

STRATEGY_DESCRIPTIONS = {
    "c1": "Splits lectures into fixed 30-second chunks. Simple baseline.",
    "c2": "Splits at word-count boundaries (≈200 words). Respects sentence structure.",
    "c3": "Splits when slide changes detected via OCR similarity. Semantic chunks. (Novel contribution)",
}

TYPE_CSS = {
    "code":        ("code-card",        "type-tag type-code",        "💻 code"),
    "theoretical": ("theory-card",      "type-tag type-theoretical", "📐 theory"),
    "conceptual":  ("conceptual-card",  "type-tag type-conceptual",  "💡 concept"),
}

INTENT_CSS = {
    "code":        "intent-badge intent-code",
    "theoretical": "intent-badge intent-theoretical",
    "conceptual":  "intent-badge intent-conceptual",
}


def _format_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def render_result_card(result: dict) -> None:
    ctype   = result.get("content_type", "conceptual")
    card_cls, tag_cls, tag_label = TYPE_CSS.get(ctype, TYPE_CSS["conceptual"])

    transcript = result.get("transcript", "").replace("\n", " ").strip()
    snippet    = transcript[:220] + ("…" if len(transcript) > 220 else "")
    link       = result.get("youtube_deep_link", result.get("youtube_url", ""))
    start      = result.get("start_sec", 0)
    end        = result.get("end_sec", 0)
    ts         = f"{_format_time(start)} → {_format_time(end)}"
    score      = result.get("retrieval_score", 0)
    strat      = result.get("chunking_strategy", "")

    html = f"""
    <div class="result-card {card_cls}">
      <div>
        <span class="rank-badge">#{result['rank']}</span>
        <span class="course-tag">{result.get('course_name', '')}</span>
        <span class="{tag_cls}">{tag_label}</span>
        {"<span class='course-tag'>🎬 code</span>" if result.get('is_code_segment') else ""}
      </div>
      <div class="lecture-title">{result.get('lecture_title', 'Unknown lecture')}</div>
      <div class="timestamp-line">⏱ {ts}  ·  Lecture {result.get('lecture_number', '?')}</div>
      <div class="snippet">"{snippet}"</div>
      {"<a class='play-btn' href='" + link + "' target='_blank'>▶ Play from here</a>" if link else ""}
      <div class="score-line">score: {score:.5f}  ·  strategy: {strat}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    # Expandable OCR + full transcript
    ocr = result.get("ocr_text", "").strip()
    full_tx = result.get("transcript", "").strip()
    if ocr or full_tx:
        with st.expander("Show slide text & full transcript"):
            if ocr and not result.get("ocr_failed"):
                st.markdown("**Slide OCR text:**")
                st.code(ocr.replace("\n---\n", "\n──────────\n"), language=None)
            if full_tx:
                st.markdown("**Full transcript:**")
                st.markdown(f"_{full_tx}_")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Renders sidebar controls and returns the config dict."""

    # ── DARK SIDEBAR CSS ─────────────────────────────────────
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;  /* deep black-blue */
        color: #e5e7eb !important;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    /* Inputs */
    section[data-testid="stSidebar"] .stSelectbox div,
    section[data-testid="stSidebar"] .stSlider,
    section[data-testid="stSidebar"] .stTextInput input {
        background-color: #1e293b !important;
        color: #f9fafb !important;
        border-radius: 6px !important;
    }

    /* Checkbox + toggle labels */
    section[data-testid="stSidebar"] label {
        color: #e5e7eb !important;
    }

    /* Divider */
    section[data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    /* Caption */
    section[data-testid="stSidebar"] .stCaption {
        color: #94a3b8 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── SIDEBAR CONTENT ──────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Search settings")

        strategy = st.selectbox(
            "Chunking strategy",
            options=list(STRATEGY_LABELS.keys()),
            format_func=lambda x: STRATEGY_LABELS[x],
            index=1,
        )
        st.caption(STRATEGY_DESCRIPTIONS[strategy])

        st.divider()

        top_k = st.slider("Results to show", 1, 10, 5)

        use_rerank = st.toggle("Cross-encoder re-ranking", value=True)

        use_llm = st.toggle("LLM query analysis (Ollama)", value=False)

        if use_llm:
            ollama_model = st.text_input(
                "Ollama model",
                value=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            )
            os.environ["OLLAMA_MODEL"] = ollama_model
            st.caption("Run: ollama serve")

        st.divider()
        st.markdown("### 📚 Course filter")
        st.caption("Leave all unchecked to search all courses.")

        courses = [
            "Introduction to Algorithms and Analysis",
            "Design and Analysis of Algorithms",
            "Deep Learning",
            "Operating Systems",
            "Database Management Systems",
            "Computer Vision",
            "Computer Architecture and Organization",
            "Machine Learning",
            "Computer Networks",
        ]

        selected_courses = []
        for c in courses:
            if st.checkbox(c, key=f"course_{c}"):
                selected_courses.append(c)

        st.divider()
        st.markdown("### ℹ️ About")
        st.caption(
            "NPTEL Lecture Retrieval — Master's Thesis\n\n"
            "Multimodal Semantic Indexing and Retrieval for NPTEL "
            "Lecture Videos using Joint Audio-Visual Embeddings."
        )

    return {
        "strategy": strategy,
        "top_k": top_k,
        "use_rerank": use_rerank,
        "use_llm": use_llm,
        "course_filter": selected_courses,
    }


# ─────────────────────────────────────────────────────────────────────────────
# History
# ─────────────────────────────────────────────────────────────────────────────

def init_history():
    if "query_history" not in st.session_state:
        st.session_state.query_history = []


def add_to_history(query: str, n_results: int, latency: float, strategy: str):
    st.session_state.query_history.insert(0, {
        "query":    query,
        "results":  n_results,
        "latency":  latency,
        "strategy": strategy,
    })
    st.session_state.query_history = st.session_state.query_history[:10]


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main():
    init_history()
    ret    = load_retriever()
    config = render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
      <div class="app-title" >🎓 NPTEL Lecture Retrieval</div>
      <div class="app-subtitle">
        multimodal semantic search · audio-visual embeddings · youtube deep links
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Query input ───────────────────────────────────────────────────────
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "Search lectures",
            placeholder="e.g.  how does binary search tree insertion work",
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    # Example queries
    st.markdown(
        "<div style='font-size:0.78rem;color:#9ca3af;margin-top:4px'>"
        "Try: &nbsp;"
        "<code>BST insertion</code> &nbsp;·&nbsp; "
        "<code>how does virtual memory work</code> &nbsp;·&nbsp; "
        "<code>python code for bubble sort</code> &nbsp;·&nbsp; "
        "<code>explain backpropagation</code>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Run search ────────────────────────────────────────────────────────
    if (search_clicked or query) and query.strip():
        with st.spinner("Searching…"):
            t0 = time.time()
            try:
                results = ret.search(
                    query      = query.strip(),
                    strategy   = config["strategy"],
                    top_k      = config["top_k"],
                    use_llm    = config["use_llm"],
                    use_rerank = config["use_rerank"],
                    verbose    = False,
                )
            except FileNotFoundError as e:
                st.error(str(e))
                return
            latency = time.time() - t0

        # Apply course filter if any selected
        if config["course_filter"]:
            results = [
                r for r in results
                if r.get("course_name") in config["course_filter"]
            ]

        add_to_history(query, len(results), latency, config["strategy"])

        # ── Results header ────────────────────────────────────────────────
        if results:
            intent     = results[0].get("query_intent", "conceptual")
            intent_cls = INTENT_CSS.get(intent, INTENT_CSS["conceptual"])
            st.markdown(
                f"<div style='margin:1rem 0 0.5rem 0;font-size:0.85rem;color:#6b7280'>"
                f"Found <b>{len(results)}</b> results in <b>{latency:.2f}s</b> &nbsp;"
                f"<span class='{intent_cls}'>intent: {intent}</span>"
                f"&nbsp;·&nbsp; strategy: <code>{config['strategy'].upper()}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.divider()

            # Two-column layout for results
            left, right = st.columns([3, 1])
            with left:
                for result in results:
                    render_result_card(result)

            with right:
                st.markdown("#### Query stats")
                st.metric("Latency", f"{latency:.2f}s")
                st.metric("Results", len(results))
                st.metric("Strategy", config["strategy"].upper())
                st.metric("Intent", intent)

                if st.session_state.query_history:
                    st.markdown("---")
                    st.markdown("#### Recent queries")
                    for h in st.session_state.query_history[:5]:
                        st.markdown(
                            f"<div style='font-size:0.75rem;padding:3px 0;"
                            f"border-bottom:1px solid #f3f4f6;color:#374151'>"
                            f"<code>{h['strategy']}</code> {h['query'][:35]}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
        else:
            st.markdown(
                "<div class='no-results'>"
                "No results found. Try different keywords or a broader query."
                "</div>",
                unsafe_allow_html=True,
            )

    else:
        # Landing state — show system info
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        for col, (sid, label, desc) in zip(
            [c1, c2, c3],
            [
                ("C1", "Fixed window",       "30-second chunks. Baseline."),
                ("C2", "Utterance window",   "Word-count boundaries. Respects sentences."),
                ("C3", "Slide boundary",     "OCR Jaccard similarity. Semantic chunks."),
            ],
        ):
            with col:
                st.markdown(
                    f"<div style='background:#f9fafb;border:1px solid #e5e7eb;"
                    f"border-radius:8px;padding:1rem;text-align:center'>"
                    f"<div style='font-size:1.4rem;font-weight:700;color:#1a1a2e'>{sid}</div>"
                    f"<div style='font-weight:600;color:#374151;font-size:0.85rem'>{label}</div>"
                    f"<div style='color:#9ca3af;font-size:0.75rem;margin-top:4px'>{desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
